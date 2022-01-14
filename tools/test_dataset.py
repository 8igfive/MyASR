import yaml
import torch.distributed as dist
import pdb
from otrans.data.wenetspeech import Dataset
from otrans.data.audio import AudioDataset
from otrans.data.loader import FeatureLoader
from otrans.data.loader import collate_fn_with_eos_bos
from torch.utils.data import DataLoader
import torchaudio as ta
import torch
import numpy as np
import random

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def normalization(feature):
    std, mean = torch.std_mean(feature)
    return (feature - mean) / std
def spec_augment(
    mel_spectrogram,
    freq_mask_num=2,
    time_mask_num=2,
    freq_mask_rate=0.3,
    time_mask_rate=0.05,
    max_mask_time_len=100):

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

    return warped_mel_spectrogram
def audio_process(wav, sample_rate):
    feature = ta.compliance.kaldi.fbank(
            wav, num_mel_bins=40,
            sample_frequency=sample_rate, dither=0.0, energy_floor=0.0
            )
    feature = normalization(feature)
    before = feature # 应该是拷贝
    feature = spec_augment(feature, 
                           freq_mask_num= 2,
                           time_mask_num= 5,
                           freq_mask_rate= 0.3,
                           time_mask_rate= 0.05)
    return before, feature

def test_wenetspeech():
    conf_path = 'scripts/conf/train_conformer_ctc_cmvn.yaml'
    with open(conf_path, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
        data_conf = configs['data']
    with open(data_conf['vocab'], 'r', encoding='utf8') as fin:
        symbol_table = dict(
            map(
                lambda x: (x[0], int(x[1])),
                filter(
                    lambda x: len(x) == 2,
                    map(
                        lambda x: x.strip().split(),
                        fin.readlines()
                    )
                )
            )
        )
    data_conf['batch_conf']['batch_size'] = 1
    data_conf['shuffle'] = False
    data_conf['sort'] = False
    data_conf['volume_perturb'] = False
    data_conf['speed_perturb'] = False
    data_conf['gaussian_noise'] = 0.0
    data_conf['normalization'] = False
    dataset = Dataset(data_conf['data_type'],
                    #   data_conf['train_path'], 
                      'dump/corpus/aishell/train/data.list', 
                      symbol_table, data_conf, 
                      bpe_model=None, 
                      partition=True, 
                      is_eval=False)
    # loader = DataLoader(dataset,
    #                     batch_size=None,
    #                     pin_memory=data_conf['pin_memory'],
    #                     num_workers=1,
    #                     prefetch_factor=data_conf['prefetch'])
    print('start test')
    iter_data = iter(dataset)
    for i in range(1):
        data = next(iter_data)
    pdb.set_trace()

def test_audio():
    conf_path = 'scripts/conf/test_conformer.yaml'
    with open(conf_path, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    dataset = AudioDataset(configs['data'], configs['data']['train'], False)
    loader = FeatureLoader(configs, 'train')
    iterLoader = iter(loader.loader)
    data = next(iterLoader)
    
    pdb.set_trace()
if __name__ == '__main__':
    # wav = torch.load('dump/temp/wav.pt')['wav']
    # # ws_fbank = torch.load('dump/temp/fbank.pt')['feat']
    # ws_b = torch.load('dump/temp/ws_before_specaug.pt')['feat']
    # ws_a = torch.load('dump/temp/ws_after_specaug.pt')['feat']
    # ws_f = torch.load('dump/temp/ws_feat.pt')['feat']
    # au_b, au_a = audio_process(wav, 16000)
    # # ws_feat = output['feat']
    # pdb.set_trace()

    test_wenetspeech()
