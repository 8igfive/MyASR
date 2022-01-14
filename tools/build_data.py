import os
import time
import pandas as pd
import cn2an


def collectReadyData(textPath, scpPath):
    with open(textPath, 'r', encoding='utf8') as fin:
        lines = list(map(lambda x: x.strip().split(maxsplit=1), fin.readlines()))
    samples = {id: [text] for id, text in lines}
    with open(scpPath, 'r', encoding='utf8') as fin:
        lines = list(map(lambda x: x.strip().split(), fin.readlines()))
    for id, wav in lines:
        if id in samples:
            samples[id].append(wav)
    print(next(iter(samples.items())))
    return samples

def modifyText(text, vocabPath):
    with open(vocabPath, 'r', encoding='utf8') as fin:
        vocab = set(map(lambda x: x.strip().split(' ')[0], fin.readlines()))
    notUse = {'，', '。', '！', '？', '、'}
    ret = list()
    for word in text:
        if word in vocab and word not in notUse:
            ret.append(word)
    return ' '.join(ret)

def collectCVCorpus(path, pathPrefix='', vocabPath=None):
    dataDF = pd.read_csv(path, encoding='utf8', sep='\t')
    samples = {'{}_{}'.format(id, i) : [modifyText(text, vocabPath), os.path.join(pathPrefix, wav)] 
                 for i, (id, text, wav) in enumerate(dataDF[['client_id', 'sentence', 'path']].iloc)}
    assert len(dataDF) == len(samples)
    print(next(iter(samples.items())))
    return samples

def dumpData(data, dirPath, name):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    text_data = ['{} {}'.format(id, text) for id, (text, _) in data.items()]
    wav_data = ['{} {}'.format(id, wav) for id, (_, wav) in data.items()]
    with open(os.path.join(dirPath, '{}_text'.format(name)), 'w', encoding='utf8') as fout:
        fout.write('\n'.join(text_data))
    with open(os.path.join(dirPath, '{}_wav.scp'.format(name)), 'w', encoding='utf8') as fout:
        fout.write('\n'.join(wav_data))

def collectDataFromName(dataDir):
    def digit2str(s):
        d2s = {
            '0': '零',
            '1': '一',
            '2': '二',
            '3': '三',
            '4': '四',
            '5': '五',
            '6': '六',
            '7': '七',
            '8': '八',
            '9': '九'
        }
        res = ''
        for c in s:
            if c in d2s:
                res += d2s[c]
            else:
                res += c
        return res
    wavs = list(
        map(
            lambda x: os.path.join(dataDir, x),
            os.listdir(dataDir)
        )
    )
    texts = list(
        map(
            lambda x: digit2str(x),
            map(
                lambda x: ' '.join(x.split('-')[0]),
                os.listdir(dataDir)
            )
        )
    )
    data = dict(enumerate(zip(texts, wavs)))
    return data

    


if __name__ == '__main__':
    # toolsDir = os.path.dirname(os.path.abspath(__file__))
    # projectDir = os.path.dirname(toolsDir)
    # textPath = os.path.join(projectDir, 'egs', 'aishell2', 'cat_text')
    # wavPath = os.path.join(projectDir, 'egs', 'aishell2', 'cat_wav.scp')
    # cvCorpusPath = os.path.join(projectDir, 'dump', 'corpus', 'cv-corpus-7.0-2021-07-21', 'zh-CN', 'train.tsv')
    # cvPathPrefix = os.path.join(projectDir, 'dump', 'corpus', 'cv-corpus-7.0-2021-07-21', 'zh-CN', 'clips')
    # vocabPath = os.path.join(projectDir, 'egs', 'aishell2', 'vocab')
    # samples0 = collectReadyData(textPath, wavPath)
    # samples1 = collectCVCorpus(cvCorpusPath, cvPathPrefix, vocabPath)
    # samples0.update(samples1)
    # TIME = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    # dumpData(samples0, os.path.join('dump', TIME), 'originalAndCV')

    dataDir = r'/home/LAB/qujy/data/label/cn'
    data = collectDataFromName(dataDir)
    dumpData(data, 'dump/corpus/data_label_cn', 'data211')