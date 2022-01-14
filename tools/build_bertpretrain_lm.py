import os
import pdb
import torch
from otrans.model.lm import BertPretrainLanguageModel


def loadVocab(path):
    with open(path, 'r', encoding='utf8') as fin:
        tokensInLine = list(map(lambda x: x.strip().split(' '), fin.read().split('\n')))
        if tokensInLine[-1][0] == '':
            tokensInLine = tokensInLine[:-1]
        # print(tokensInLine[-1])
    token2idx = dict()
    for idx, tokenInLine in enumerate(tokensInLine):
        if tokenInLine[0] not in token2idx:
            token2idx[tokenInLine[0]] = idx
    idx2token = dict()
    for token, idx in token2idx.items():
        idx2token[idx] = token
    return token2idx, idx2token

def getSpecialTokenMap():
    V2L = {
        '<PAD>': '[PAD]',
        '<S/E>': '[CLS]',
        '<UNK>': '[UNK]'
    }
    L2V = {
        '[PAD]': '<PAD>',
        '[CLS]': '<S/E>',
        '[SEP]': '<S/E>',
        '[UNK]': '<UNK>'
    }
    return V2L, L2V

def getSpecialToken(name, modelType):
    temp = {
        'voice': {
            'PAD': '<PAD>',
            'START': '<S/E>',
            'END': '<S/E>',
            'UNK': '<UNK>'
        },
        'lm': {
            'PAD': '[PAD]',
            'START': '[CLS]',
            'END': '[SEP]',
            'UNK': '[UNK]'
        }
    }
    return temp[modelType][name]

def getConvertMatrix(voiceVocabPath, lmVocabPath, convert2UNK=False):
    voiceT2I, voiceI2T = loadVocab(voiceVocabPath)
    lmT2I, lmI2T = loadVocab(lmVocabPath)
    voiceVocabSize = len(voiceT2I)
    lmVocabSize = len(lmT2I)
    V2L, L2V = getSpecialTokenMap()
    voice2lm = torch.zeros(size=(voiceVocabSize, lmVocabSize))
    lm2voice = torch.zeros(size=(lmVocabSize, voiceVocabSize))
    for voiceToken, voiceIdx in voiceT2I.items():
        if voiceToken in V2L:
            lmToken = V2L[voiceToken]
        elif voiceToken in lmT2I:
            lmToken = voiceToken
        else:
            lmToken = getSpecialToken('UNK', 'lm')
        lmIdx = lmT2I[lmToken]
        voice2lm[voiceIdx, lmIdx] = 1
        lm2voice[lmIdx, (voiceIdx if lmToken != getSpecialToken('UNK', 'lm') else voiceT2I[getSpecialToken('UNK', 'voice')])] = 1
    for lmToken in L2V:
        voiceToken = L2V[lmToken]
        lmIdx = lmT2I[lmToken]   
        voiceIdx = voiceT2I[voiceToken]
        lm2voice[lmIdx, voiceIdx] = 1
    if convert2UNK:
        voiceUNKIdx = voiceT2I[getSpecialToken('UNK', 'voice')]
        for lmToken, lmIdx in lmT2I.items():
            if lmToken not in voiceT2I and lmToken not in L2V:
                lm2voice[lmIdx, voiceUNKIdx] = 1
    return voice2lm, lm2voice

    


def buildBertpretrainLm(voiceVocabPath, lmVocabPath, savePath, modelPath, convert2UNK=False):
    params = dict()
    params['model'] = dict()
    params['model']['type'] = 'bertpretrain_lm'

    voice2lm, lm2voice = getConvertMatrix(voiceVocabPath, lmVocabPath, convert2UNK)

    params['model']['voice2lm'] = voice2lm
    params['model']['lm2voice'] = lm2voice
    params['model']['voiceVocabSize'] = voice2lm.shape[0]
    params['model']['lmVocabSize'] = lm2voice.shape[0]
    # FIXME: paddding Ids : 为第二句话预测被遮蔽的词
    # params['model']['paddingIds'] = torch.tensor([711, 5018,  753, 1368, 6413, 7564, 3844, 6158, 6902, 5929, 4638, 6404,]) # roberta
    params['model']['paddingIds'] = torch.tensor([13, 131, 177, 1056, 543, 695, 558, 171, 2450, 2535, 5, 796,]) # ernie
    # FIXME: MASK Ids
    # params['model']['maskIds'] = torch.tensor([103]) # roberta
    params['model']['maskIds'] = torch.tensor([3]) # ernie
    params['model']['modelPath'] = modelPath
    bertPretrainLM = BertPretrainLanguageModel(params['model'])
    bertPretrainLM.save_checkpoint(params, savePath)


if __name__ == '__main__':
    projectDir = os.path.dirname(os.path.abspath(__file__))
    dumpDir = os.path.join(projectDir, 'dump')
    # saveDir = os.path.join(dumpDir, 'bertpretrain_lm')      # FIXME: the path to save the model
    saveDir = os.path.join(dumpDir, 'ernie_lm')
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    voiceVocabPath = os.path.join(projectDir, 'egs', 'aishell2', 'vocab')
    # lmVocabPath = os.path.join(saveDir, 'vocab.txt')        # FIXME: the path of lm vocab
    lmVocabPath = '/home/LAB/chensi/data/Models-PLMs/ernie-1.0/vocab.txt'
    # modelPath = os.path.join(dumpDir, 'bert-base-chinese')  # FIXME: the path of pretrained model
    modelPath = 'nghuyong/ernie-1.0'
    modelName = 'ernie-1.0'                             # FIXME: the name of pretrained model
    buildBertpretrainLm(voiceVocabPath, lmVocabPath, os.path.join(saveDir, modelName), modelPath)

    # voiceT2I, _ = loadVocab(voiceVocabPath)
    # lmT2I, _ = loadVocab(lmVocabPath)
    # pdb.set_trace()
    # print(len(voiceT2I), len(lmT2I))
    # # for token, idx in voiceT2I.items():
    # #     if '' == token:
    # #         print("contain empty in voiceT2I, idx={}".format(idx))
    # # for token, idx in lmT2I.items():
    # #     if '' == token:
    # #         print("contain empty in lmT2I, idx={}".format(idx))
    # missCount = 0
    # for voiceToken in voiceT2I:
    #     if voiceToken not in lmT2I:
    #         missCount += 1
    # print(missCount)