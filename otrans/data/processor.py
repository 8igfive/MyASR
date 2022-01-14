# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import json
import random
import re
import tarfile
import pdb
from subprocess import PIPE, Popen
from urllib.parse import urlparse
import numpy as np

from otrans.data import EOS, PAD, BOS, UNK_TOKEN

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.nn.utils.rnn import pad_sequence

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])

def url_opener(data):
    """ Give url or local file, return file descriptor
        Inplace operation.

        Args:
            data(Iterable[str]): url or local file list

        Returns:
            Iterable[{src, stream}]
    """
    for sample in data:
        assert 'src' in sample
        # TODO(Binbin Zhang): support HTTP
        url = sample['src']
        try:
            pr = urlparse(url)
            # local file
            if pr.scheme == '' or pr.scheme == 'file':
                stream = open(url, 'rb')
            # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
            else:
                cmd = f'curl -s -L {url}'
                process = Popen(cmd, shell=True, stdout=PIPE)
                sample.update(process=process)
                stream = process.stdout
            sample.update(stream=stream)
            yield sample
        except Exception as ex:
            logging.warning('Failed to open {}'.format(url))

def tar_file_and_group(data):
    """ Expand a stream of open tar files into a stream of tar file contents.
        And groups the file with same prefix

        Args:
            data: Iterable[{src, stream}]

        Returns:
            Iterable[{key, wav, txt, sample_rate}]
    """
    for sample in data:
        assert 'stream' in sample
        stream = tarfile.open(fileobj=sample['stream'], mode="r|*")
        prev_prefix = None
        example = {}
        valid = True
        for tarinfo in stream:
            name = tarinfo.name
            pos = name.rfind('.')
            assert pos > 0
            prefix, postfix = name[:pos], name[pos + 1:]
            if prev_prefix is not None and prefix != prev_prefix:
                example['key'] = prev_prefix
                if valid:
                    yield example
                example = {}
                valid = True
            file_obj = stream.extractfile(tarinfo)
            try:
                if postfix == 'txt':
                    example['txt'] = file_obj.read().decode('utf8').strip()
                elif postfix in AUDIO_FORMAT_SETS:
                    waveform, sample_rate = torchaudio.load(file_obj)
                    example['wav'] = waveform
                    example['sample_rate'] = sample_rate
                else:
                    example[postfix] = file_obj.read()
            except Exception as ex:
                valid = False
                logging.warning('error to parse {}'.format(name))
            prev_prefix = prefix
        if prev_prefix is not None:
            example['key'] = prev_prefix
            yield example
        stream.close()
        if 'process' in sample:
            sample['process'].communicate()
        sample['stream'].close()

def parse_raw(data):
    """ Parse key/wav/txt from json line

        Args:
            data: Iterable[str], str is a json line has key/wav/txt

        Returns:
            Iterable[{key, wav, txt, sample_rate}]
    """
    for sample in data:
        assert 'src' in sample
        json_line = sample['src']
        obj = json.loads(json_line)
        assert 'key' in obj
        assert 'wav' in obj
        assert 'txt' in obj
        key = obj['key']
        wav_file = obj['wav']
        txt = obj['txt']
        try:
            if 'start' in obj:
                assert 'end' in obj
                sample_rate = torchaudio.backend.sox_io_backend.info(
                    wav_file).sample_rate
                start_frame = int(obj['start'] * sample_rate)
                end_frame = int(obj['end'] * sample_rate)
                waveform, _ = torchaudio.backend.sox_io_backend.load(
                    filepath=wav_file,
                    num_frames=end_frame - start_frame,
                    frame_offset=start_frame)
            else:
                waveform, sample_rate = torchaudio.load(wav_file)
            example = dict(key=key,
                           txt=txt,
                           wav=waveform,
                           sample_rate=sample_rate)
            yield example
        except Exception as ex:
            logging.warning('Failed to read {}'.format(wav_file))

def filter(data,
           max_length=10240,
           min_length=10,
           token_max_length=200,
           token_min_length=1,
           min_output_input_ratio=0.0005,
           max_output_input_ratio=1):
    """ Filter sample according to feature and label length
        Inplace operation.

        Args::
            data: Iterable[{key, wav, label, sample_rate}]
            max_length: drop utterance which is greater than max_length(10ms)
            min_length: drop utterance which is less than min_length(10ms)
            token_max_length: drop utterance which is greater than
                token_max_length, especially when use char unit for
                english modeling
            token_min_length: drop utterance which is
                less than token_max_length
            min_output_input_ratio: minimal ration of
                token_length / feats_length(10ms)
            max_output_input_ratio: maximum ration of
                token_length / feats_length(10ms)

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'label' in sample
        # sample['wav'] is torch.Tensor, we have 100 frames every second
        num_frames = sample['wav'].size(1) / sample['sample_rate'] * 100
        if num_frames < min_length:
            continue
        if num_frames > max_length:
            continue
        if len(sample['label']) < token_min_length:
            continue
        if len(sample['label']) > token_max_length:
            continue
        if num_frames != 0:
            if len(sample['label']) / num_frames < min_output_input_ratio:
                continue
            if len(sample['label']) / num_frames > max_output_input_ratio:
                continue 
        yield sample

def resample(data, resample_rate=16000):
    """ Resample data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            resample_rate: target resample rate

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        if sample_rate != resample_rate:
            sample['sample_rate'] = resample_rate
            sample['wav'] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate)(waveform)
        yield sample

def speed_perturb(data, speeds=None):
    """ Apply speed perturb to the data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            speeds(List[float]): optional speed

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    if speeds is None:
        speeds = [0.9, 1.0, 1.1]
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        speed = random.choice(speeds)
        if speed != 1.0:
            wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate,
                [['speed', str(speed)], ['rate', str(sample_rate)]])
            sample['wav'] = wav
        yield sample

def volume_perturb(data):
    for sample in data:
        assert 'wav' in sample
        wavform = sample['wav']
        volume_factor = 10 ** (random.uniform(-2, 2) / 20)
        wavform *= volume_factor
        sample['wav'] = wavform
        yield sample

def compute_fbank(data,
                  num_mel_bins=23,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0):
    """ Extract fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'key' in sample
        assert 'label' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        # waveform = waveform * (1 << 15) # FIXME: perturb in volume_perturb
        # Only keep key, feat, label
        mat = kaldi.fbank(waveform,
                          num_mel_bins=num_mel_bins,
                          frame_length=frame_length,
                          frame_shift=frame_shift,
                          dither=dither,
                        #   energy_floor=0.0,
                          sample_frequency=sample_rate)
        yield dict(key=sample['key'], label=sample['label'], feat=mat)

def normalization(data):
    for sample in data:
        assert 'feat' in sample
        feat = sample['feat']
        std, mean = torch.std_mean(feat)
        feat = (feat - mean) / std
        sample['feat'] = feat
        yield sample

def cmvn(data, global_cmvn):
    global_mean = torch.from_numpy(np.load(f'{global_cmvn}.mean.npy'))
    global_std = torch.from_numpy(np.load(f'{global_cmvn}.std.npy'))
    
    for sample in data:
        assert 'feat' in sample
        if global_mean.dtype != sample['feat'].dtype:
            global_mean = global_mean.to(sample['feat'].dtype)
            global_std = global_std.to(sample['feat'].dtype)
        sample['feat'] = (sample['feat'] - global_mean) / global_std
        yield sample

def gaussian_noise(data, gaussian_noise=0.0):
    for sample in data:
        assert 'feat' in sample
        feat = sample['feat']
        noise = torch.normal(torch.zeros(feat.size(-1)), std=gaussian_noise)
        sample['feat'] = feat + noise
        yield sample

def tokenize(data, symbol_table, bpe_model=None):
    """ Decode text to chars or BPE
        Inplace operation

        Args:
            data: Iterable[{key, wav, txt, sample_rate}]

        Returns:
            Iterable[{key, wav, txt, tokens, label, sample_rate}]
    """
    # TODO(Binbin Zhang): Support BPE
    if bpe_model is not None:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model)
    for sample in data:
        assert 'txt' in sample
        txt = sample['txt']
        label = []
        tokens = []
        if bpe_model is not None:
            txt = bpe_preprocess(txt)
            mix_chars = seg_char(txt)
            for j in mix_chars:
                for k in j.strip().split("▁"):
                    if not k.encode('UTF-8').isalpha():
                        tokens.append(k)
                    else:
                        for l in sp.encode_as_pieces(k):
                            tokens.append(l)
        else:
            for ch in txt:
                if ch == ' ':
                    ch = "▁"
                tokens.append(ch)

        for ch in tokens:
            if ch in symbol_table:
                label.append(symbol_table[ch])
            elif '<unk>' in symbol_table:
                label.append(symbol_table[UNK_TOKEN])

        sample['tokens'] = tokens
        sample['label'] = label
        yield sample

def bpe_preprocess(text):
    """ Use ▁ for blank among english words
        Warning: it is "▁" symbol, not "_" symbol
    """
    text = text.upper()
    text = re.sub(r'([A-Z])[ ]([A-Z])', r'\1▁\2', text)
    text = re.sub(r'([A-Z])[ ]([A-Z])', r'\1▁\2', text)
    text = text.replace(' ', '')
    text = text.replace('\xEF\xBB\xBF', '')
    return text

def seg_char(text):
    pattern = re.compile(r'([\u4e00-\u9fa5])')
    chars = pattern.split(text)
    chars = [w for w in chars if len(w.strip()) > 0]
    return chars

def spec_aug(data, num_t_mask=2, num_f_mask=2, max_t=50, max_f=10, max_w=80):
    """ Do spec augmentation
        Inplace operation

        Args:
            data: Iterable[{key, feat, label}]
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask
            max_w: max width of time warp

        Returns
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'feat' in sample
        x = sample['feat']
        assert isinstance(x, torch.Tensor)
        y = x.clone().detach()
        max_frames = y.size(0)
        max_freq = y.size(1)
        # time mask
        for i in range(num_t_mask):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            y[start:end, :] = 0
        # freq mask
        for i in range(num_f_mask):
            start = random.randint(0, max_freq - 1)
            length = random.randint(1, max_f)
            end = min(max_freq, start + length)
            y[:, start:end] = 0
        sample['feat'] = y
        yield sample

def my_spec_aug(
    data,
    freq_mask_num=2,
    time_mask_num=2,
    freq_mask_rate=0.3,
    time_mask_rate=0.05,
    max_mask_time_len=100):

    for sample in data:
        assert 'feat' in sample
        mel_spectrogram = sample['feat']
        tau = mel_spectrogram.shape[0]
        v = mel_spectrogram.shape[1]

        warped_mel_spectrogram = mel_spectrogram

        freq_masking_para = int(v * freq_mask_rate)
        time_masking_para = min(int(tau * time_mask_rate), max_mask_time_len)

        # Step 1 : Frequency masking
        if freq_mask_num > 0:
            for _ in range(freq_mask_num):
                f = np.random.uniform(low=0.0, high=freq_masking_para)
                f = int(f)
                f0 = random.randint(0, v-f)
                warped_mel_spectrogram[:, f0:f0+f] = 0
        
        # Step 2 : Time masking
        if time_mask_num > 0:
            for _ in range(time_mask_num):
                t = np.random.uniform(low=0.0, high=time_masking_para)
                t = int(t)
                t0 = random.randint(0, tau-t)
                warped_mel_spectrogram[t0:t0+t, :] = 0
        
        sample['feat'] = warped_mel_spectrogram
        yield sample



def shuffle(data, shuffle_size=10000):
    """ Local shuffle the data

        Args:
            data: Iterable[{key, feat, label}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{key, feat, label}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x

def sort(data, sort_size=500):
    """ Sort the data by feature length.
        Sort is used after shuffle and before batch, so we can group
        utts with similar lengths into a batch, and `sort_size` should
        be less than `shuffle_size`

        Args:
            data: Iterable[{key, feat, label}]
            sort_size: buffer size for sort

        Returns:
            Iterable[{key, feat, label}]
    """

    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= sort_size:
            buf.sort(key=lambda x: x['feat'].size(0))
            for x in buf:
                yield x
            buf = []
    # The sample left over
    buf.sort(key=lambda x: x['feat'].size(0))
    for x in buf:
        yield x

def static_batch(data, batch_size=16):
    """ Static batch the data by `batch_size`

        Args:
            data: Iterable[{key, feat, label}]
            batch_size: batch size

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf

def dynamic_batch(data, max_frames_in_batch=12000):
    """ Dynamic batch the data until the total frames in batch
        reach `max_frames_in_batch`

        Args:
            data: Iterable[{key, feat, label}]
            max_frames_in_batch: max_frames in one batch

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    longest_frames = 0
    for sample in data:
        assert 'feat' in sample
        assert isinstance(sample['feat'], torch.Tensor)
        new_sample_frames = sample['feat'].size(0)
        longest_frames = max(longest_frames, new_sample_frames)
        frames_after_padding = longest_frames * (len(buf) + 1)
        if frames_after_padding > max_frames_in_batch:
            yield buf
            buf = [sample]
            longest_frames = new_sample_frames
        else:
            buf.append(sample)
    if len(buf) > 0:
        yield buf

def batch(data, batch_type='static', batch_size=16, max_frames_in_batch=12000):
    """ Wrapper for static/dynamic batch
    """
    if batch_type == 'static':
        return static_batch(data, batch_size)
    elif batch_type == 'dynamic':
        return dynamic_batch(data, max_frames_in_batch)
    else:
        logging.fatal('Unsupported batch type {}'.format(batch_type))

def padding(data, withEOSandBOS=True):
    """ Padding the data into training data

        Args:
            data: Iterable[List[{key, feat, label}]]

        Returns:
            Iterable[Tuple(keys, feats, labels, feats lengths, label lengths)]
    """
    for sample in data:
        assert isinstance(sample, list)
        feats_length = torch.tensor([x['feat'].size(0) for x in sample],
                                    dtype=torch.int32)
        order = torch.argsort(feats_length, descending=True)
        feats_lengths = torch.tensor(
            [sample[i]['feat'].size(0) for i in order], dtype=torch.int32)
        sorted_feats = [sample[i]['feat'] for i in order]
        sorted_keys = [sample[i]['key'] for i in order]
        if withEOSandBOS:
            sorted_labels = [
                torch.tensor([BOS] + sample[i]['label'] + [EOS], dtype=torch.int64) for i in order
            ]
            label_lengths = torch.tensor([x.size(0) - 1 for x in sorted_labels],
                                        dtype=torch.int32) # 原标签长度 + 1
        else:
            sorted_labels = [
                torch.tensor(sample[i]['label'], dtype=torch.int64) for i in order
            ]
            label_lengths = torch.tensor([x.size(0) for x in sorted_labels],
                                        dtype=torch.int32) # 原标签长度 + 1

        padded_feats = pad_sequence(sorted_feats,
                                    batch_first=True,
                                    padding_value=0)
        max_f_length = max(feats_lengths)
        padded_feat_mask = torch.tensor([
            [True] * f_length + [False] * (max_f_length - f_length) for f_length in feats_lengths
        ], dtype=torch.bool)

        padding_labels = pad_sequence(sorted_labels,
                                      batch_first=True,
                                      padding_value=PAD)
        padding_label_mask = padding_labels != PAD
        
        inputs = {
            'inputs': padded_feats,
            'inputs_length': feats_lengths,
            'mask': padded_feat_mask
        }
        targets = {
            'targets': padding_labels,
            'targets_length': label_lengths,
            'mask': padding_label_mask
        }
        # yield (sorted_keys, padded_feats, padding_labels, feats_lengths,
        #        label_lengths)
        yield sorted_keys, inputs, targets