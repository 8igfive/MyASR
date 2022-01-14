import os
from tqdm import tqdm

def clean_log(path):
    with open(path, 'r', encoding='utf8') as fin:
        lines = fin.readlines()
    remain = list()
    for line in tqdm(lines):
        if len(line) < 60 or line[59] != '.':
            remain.append(line)
    splitPath = list(os.path.split(path))
    splitPath[-1] = 'clean_' + splitPath[-1]
    with open(os.path.join(*splitPath), 'w', encoding='utf8') as fout:
        fout.write('\n'.join(remain))

if __name__ == '__main__':
    path = os.path.join('dump', 'corpus', 'WenetSpeech', 'download.log')
    clean_log(path)
