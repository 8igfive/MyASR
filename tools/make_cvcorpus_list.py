import argparse
import io
import logging
import os
import tarfile
import time
import multiprocessing
import pandas
from pandas.core.arrays import sparse

import torch
import torchaudio
import torchaudio.backend.sox_io_backend as sox

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])

def write_tar_file(data_list,
                   no_segments,
                   tar_file,
                   resample=16000,
                   index=0,
                   total=1):
    logging.info('Processing {} {}/{}'.format(tar_file, index, total))
    read_time = 0.0
    save_time = 0.0
    write_time = 0.0
    with tarfile.open(tar_file, "w") as tar:
        prev_wav = None
        for item in data_list:
            if no_segments:
                key, txt, wav = item
            else:
                key, txt, wav, start, end = item

            suffix = wav.split('.')[-1]
            assert suffix in AUDIO_FORMAT_SETS
            if no_segments:
                # ts = time.time()
                # with open(wav, 'rb') as fin:
                #     data = fin.read()
                # read_time += (time.time() - ts)
                if wav != prev_wav:
                    ts = time.time()
                    waveforms, sample_rate = sox.load(wav, normalize=False)
                    read_time += (time.time() - ts)
                    prev_wav = wav
                audio = waveforms[:1, :]
                if sample_rate != resample:
                    audio = torchaudio.transforms.Resample(
                        sample_rate, resample)(audio)
                ts = time.time()
                f = io.BytesIO()
                sox.save(f, audio, resample, format="wav", bits_per_sample=16)
                # Save to wav for segments file
                suffix = "wav"
                f.seek(0)
                data = f.read()
                save_time += (time.time() - ts)
            else:
                if wav != prev_wav:
                    ts = time.time()
                    waveforms, sample_rate = sox.load(wav, normalize=False)
                    read_time += (time.time() - ts)
                    prev_wav = wav
                start = int(start * sample_rate)
                end = int(end * sample_rate)
                audio = waveforms[:1, start:end]

                # resample
                if sample_rate != resample:
                    audio = torchaudio.transforms.Resample(
                        sample_rate, resample)(audio)

                ts = time.time()
                f = io.BytesIO()
                sox.save(f, audio, resample, format="wav", bits_per_sample=16)
                # Save to wav for segments file
                suffix = "wav"
                f.seek(0)
                data = f.read()
                save_time += (time.time() - ts)

            assert isinstance(txt, str)
            ts = time.time()
            txt_file = key + '.txt'
            txt = txt.encode('utf8')
            txt_data = io.BytesIO(txt)
            txt_info = tarfile.TarInfo(txt_file)
            txt_info.size = len(txt)
            tar.addfile(txt_info, txt_data)

            wav_file = key + '.' + suffix
            wav_data = io.BytesIO(data)
            wav_info = tarfile.TarInfo(wav_file)
            wav_info.size = len(data)
            tar.addfile(wav_info, wav_data)
            write_time += (time.time() - ts)
        logging.info('read {} save {} write {}'.format(read_time, save_time,
                                                       write_time))

def modifyText(text, vocab):
    notUse = {'，', '。', '！', '？', '、'}
    ret = list()
    for word in text:
        if word in vocab and word not in notUse:
            ret.append(word)
    return ''.join(ret)

if __name__ == '__main__':
    file_path = r'dump/corpus/cv-corpus-7.0-2021-07-21/zh-CN/train.tsv'
    samples = pandas.read_csv(file_path, sep='\t')
    vocabPath = 'egs/aishell2/vocab'
    with open(vocabPath, 'r', encoding='utf8') as fin:
        vocab = set(map(lambda x: x.strip().split(' ')[0], fin.readlines()))
        assert '' not in vocab
    data = list()
    path_prefix = 'dump/corpus/cv-corpus-7.0-2021-07-21/zh-CN/clips'
    for sample in samples.iloc:
        key = sample['path'].split('.')[0]
        txt = modifyText(sample['sentence'], vocab)
        wav = os.path.join(path_prefix, sample['path'])
        data.append((key, txt, wav))
    
    num = 1000
    processes_num = 16
    shards_dir = 'dump/corpus/cv-corpus-7.0-2021-07-21/train'
    prefix = 'shards'
    resample = 16000
    shards_list_path = 'dump/corpus/cv-corpus-7.0-2021-07-21/train/data.list'
    no_segments = True

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    chunks = [data[i: i + num] for i in range(0, len(data), num)]
    pool = multiprocessing.Pool(processes=processes_num)
    shards_list = []
    tasks_list = []
    num_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        tar_file = os.path.join(shards_dir,
                                '{}_{:09d}.tar'.format(prefix, i))
        shards_list.append(tar_file)
        pool.apply_async(
            write_tar_file,
            (chunk, no_segments, tar_file, resample, i, num_chunks))

    pool.close()
    pool.join()

    with open(shards_list_path, 'w', encoding='utf8') as fout:
        for name in shards_list:
            fout.write(name + '\n')



