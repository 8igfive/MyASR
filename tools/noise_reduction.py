import argparse
from genericpath import exists
import os
import time
import re
from tqdm import tqdm
import numpy as np
from scipy.io import wavfile
from wiener_scalart import wienerScalart

TIME = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKPLACE_DIR = os.path.dirname(CURRENT_DIR)
DUMP_DIR = os.path.join(WORKPLACE_DIR, os.path.join('dump', TIME))
DUMP_FEAT = 'feat_{}.scp'.format(TIME)
DUMP_TEXT = 'text_{}'.format(TIME)
FEAT_FORMAT = r'\s?(.+?)\s+(.+?\.wav)'
intMap = {np.dtype('int8') : (0x7f, -0x80), 
          np.dtype('int16') : (0x7fff, -0x8000), 
          np.dtype('int32') : (0x7fffffff, -0x8000000), 
          np.dtype('int64') : (0x7fffffffffffffff, -0x8000000000000000)}

def noise_reduct(args, filePath, dumpPath):
    sampleRate, musicData = wavfile.read(filePath)
    dataType = np.dtype('int16')
    musicData.dtype = dataType # FIXME: wavfile 读取的结果数据类型可能有问题
    if args.debug:
        print(min(musicData), max(musicData), intMap[dataType][0] + 1)
    if dataType in intMap:
        musicData = musicData / (intMap[dataType][0] + 1)
        if args.debug:
            print(min(musicData), max(musicData))
    
    newData = wienerScalart(musicData, sampleRate)

    if dataType in intMap:
        if args.debug:
            print(min(newData), max(newData))
        newData = newData * (intMap[dataType][0])
        newData = newData.astype(dataType)
    if args.debug:
        print(max(newData), min(newData))
    
    wavfile.write(dumpPath, sampleRate, newData)


def main(args):
    if args.feat is None or args.text is None:
        print('lack of feat file or text file')
        return
    if os.path.abspath(args.dumpFeat) != args.dumpFeat:
        args.dumpFeat = os.path.join(DUMP_DIR, args.dumpFeat)
    if os.path.abspath(args.dumpText) != args.dumpText:
        args.dumpText = os.path.join(DUMP_DIR, args.dumpText)
    
    if not os.path.exists(DUMP_DIR):
        os.makedirs(DUMP_DIR)
    
    
    with open(args.feat, 'r', encoding='utf8') as f:
        dataPairs = re.findall(FEAT_FORMAT, f.read())
    with open(args.dumpFeat, 'w', encoding='utf8') as f:
        for i in tqdm(range(len(dataPairs))):
            dataPair = dataPairs[i]
            pathList = os.path.split(dataPair[1])
            dumpPath = os.path.join(args.dumpDir, pathList[-1])
            f.write('{} {}\n'.format(dataPair[0], dumpPath))
            noise_reduct(args, dataPair[1], dumpPath)
    with open(args.text, 'r', encoding='utf8') as fin:
        with open(args.dumpText, 'w', encoding='utf8') as fout:
            fout.write(fin.read())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--feat', type=str, default=None, help='feat file path')
    parser.add_argument('-t', '--text', type=str, default=None, help='text file path')
    parser.add_argument('-dd', '--dumpDir', type=str, default=DUMP_DIR, help='the directory where holds new .wav files')
    parser.add_argument('-df', '--dumpFeat', type=str, default=os.path.join(DUMP_DIR, DUMP_FEAT), help='dump feat file path')
    parser.add_argument('-dt', '--dumpText', type=str, default=os.path.join(DUMP_DIR, DUMP_TEXT), help='dump text file path')
    parser.add_argument('-n', '--noiseLength', type=float, default=0.25, help='the noise time length at the beggining of the audio')
    parser.add_argument('-db', '--debug', action='store_true', help='print debug message')

    args = parser.parse_args()

    main(args)