# MyASR

This is a project (or personal test) based on [OpenTransformer](https://github.com/ZhengkunTian/OpenTransformer)

# Requirements
Pytorch >= 1.2.0 (<= 1.6.0)

Torchaudio >= 0.3.0

## Function

- Speech Transformer / Conformer

- Label Smoothing

- Tie Weights of Embedding with output softmax layer

- Data Augmentation([SpecAugument](https://arxiv.org/abs/1904.08779))

- Extract Fbank features in a online fashion

- Read the feature with the kaldi or espnet format!

- Visualization based Tensorboard [Will Be Updated Soon!]

- Batch Beam Search with Length Penalty

- Multiple Optimizers and Schedulers

- Multiple Activation Functions in FFN

- Multi GPU 

- LM Shollow Fusion

## Prepare
vocab
```
# character idx
<PAD> 0
<S/E> 1
<UNK> 2
我 3
你 4
...
```
character
```
BAC009S0764W0139 国 家 统 计 局 的 数 据 显 示
BAC009S0764W0140 其 中 广 州 深 圳 甚 至 出 现 了 多 个 日 光 盘
BAC009S0764W0141 零 三 年 到 去 年
BAC009S0764W0142 市 场 基 数 已 不 可 同 日 而 语
BAC009S0764W0143 在 市 场 整 体 从 高 速 增 长 进 入 中 高 速 增 长 区 间 的 同 时
BAC009S0764W0144 一 线 城 市 在 价 格 较 高 的 基 础 上 整 体 回 升 并 领 涨 全 国
BAC009S0764W0145 绝 大 部 分 三 线 城 市 房 价 仍 然 下 降
BAC009S0764W0146 一 线 楼 市 成 交 量 激 增
BAC009S0764W0147 三 四 线 城 市 依 然 冷 清
```
if you want to compute features online, please make sure you have a wav.scp file.
```
# wav.scp
# id path
BAC009S0764W0139 /data/aishell/wav/BAC009S0764W0139.wav
```

## Train
- Single GPU
```python
python run.py -c egs/aishell/conf/transformer.yaml
```
- Multi GPU Training based DataParallel
```python
python run.py -c egs/aishell/transformer.yaml -n 2 -g 0,1
```

## Average the parameters of the last N epochs
```python
python tools/average.py your_model_expdir 50 59    #   average the models from 50-th epoch to 59-th epoch
```

## Eval
```python
python eval.py -m model.pt
```

## Acknowledge
OpenTransformer refer to [OpenTransformer](https://github.com/ZhengkunTian/OpenTransformer).
