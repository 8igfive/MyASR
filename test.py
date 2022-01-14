import os
import sys
import pandas as pd
import pdb
import torch
from otrans.train.scheduler import BuildOptimizer, BuildScheduler

import time

def showCVCorpus():
    dirPath = os.path.join('dump', 'corpus', 'cv-corpus-7.0-2021-07-21', 'zh-CN')
    filePath = os.path.join(dirPath, 'test.tsv')
    testDF = pd.read_csv(filePath, sep='\t')
    for i, j in testDF[['client_id', 'path']].loc:
        print(i, j)
        break

def showWenetTopKLines(k=20):
    filePath = os.path.join('dump', 'corpus', 'WenetSpeech', 'DATA_UNTAR_DIRECTORY', 'WenetSpeech.json')
    with open(filePath, 'r', encoding='utf8') as fin:
        lines = list()
        for i in range(k):
            lines.append(fin.readline())
    print('\n'.join(lines))

if __name__ == '__main__':
    # showWenetTopKLines(50)
    
    # optim_checkpoint_path = 'egs/aishell2/exp/simple_transformer/latest_optimizer.pt'
    # ochkpt = torch.load(optim_checkpoint_path, map_location=torch.device('cpu'))
    # optimizer = BuildOptimizer['adam'](
    #     torch.nn.Linear(10, 256).parameters(), lr=0.001, betas=[0.9, 0.98],eps=1.0e-9, weight_decay=1.0e-6, amsgrad=False
    # )
    # scheduler = BuildScheduler['transformer'](optimizer, model_size=256, warmup_steps=12000, factor=1.0)
    # pdb.set_trace() # FIXME
    # # optimizer.load_state_dict(ochkpt['optim'])
    # global_step = ochkpt['global_step'] if 'global_step' in ochkpt else 0
    # scheduler.global_step = global_step
    # scheduler.set_lr()
    os.environ['CUDA_VISIBLE_DEVICES'] = '3,2,1,0'
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0))
    # print(torch.cuda.get_device_name(1), torch.cuda.get_device_properties(1))
    torch.cuda.set_device(1)
    t = torch.rand(10000, 10000).cuda()
    print(t.device)
    time.sleep(1000)
