import os
import random

def make_segments_wav_scp(src_scp_path, segments_path, dst_scp_path):
    with open(src_scp_path, 'r', encoding='utf8') as fin:
        wavkey2path = dict(
            map(
                lambda x: x.strip().split(),
                fin.readlines()
            )
        )
    with open(segments_path, 'r', encoding='utf8') as fin:
        scpsegments = list(
            map(
                lambda x: '{} {},{},{}'.format(x[0], wavkey2path[x[1]], x[2], x[3]),
                filter(
                    lambda x: x[1] in wavkey2path, # segKey, wavKey, start, end
                    map(
                        lambda x: x.strip().split(),
                        fin.readlines()
                    )
                )
            )
        )
    with open(dst_scp_path, 'w', encoding='utf8') as fout:
        fout.write('\n'.join(scpsegments))
    print('{} contain {} segments'.format(dst_scp_path, len(scpsegments)))
    print('segments are like: {}'.format(scpsegments[0]))
    

def sample(src_path, dst_path, devide_factor=None, num=50000):
    with open(src_path, 'r', encoding='utf8') as fin:
        srclines = list(map(lambda x: x.strip(), fin.readlines()))
    if devide_factor:
        dst_size = len(srclines) // devide_factor
    else:
        dst_size = num
    random.shuffle(srclines)
    dstlines = srclines[:dst_size]
    print('srclines={}, dstlines={}'.format(len(srclines), len(dstlines)))
    # print(dstlines[:10])
    with open(dst_path, 'w', encoding='utf8') as fout:
        fout.write('\n'.join(dstlines))

def count_lines(path):
    with open(path, 'r', encoding='utf8') as fin:
        lines = list(map(lambda x: x.strip(), fin.readlines()))
        # lines = fin.read().split('\n')
        print(lines[:3])
    return len(lines)

if __name__ == '__main__':
    # src_scp_path = '/home/LAB/qujy/open/wenet-main/examples/wenetspeech/s0/data/train_l/wav.scp'
    # segments_path = '/home/LAB/qujy/open/wenet-main/examples/wenetspeech/s0/data/train_l/segments'
    # dst_scp_path = 'egs/wenetspeech/data/train/wav.scp.segments'
    
    # make_segments_wav_scp(src_scp_path, segments_path, dst_scp_path)

    src_path = 'egs/wenetspeech/data/train/wav.scp.segments'
    dst_path = 'egs/wenetspeech/data/train/wav.scp.segments.sample_50000'
    
    sample(src_path, dst_path, num=50000)
    # count_lines(src_scp_path)
    # print(count_lines(dst_path))
