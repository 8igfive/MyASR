import json
import numpy as np
import pdb

DEBUG = True

def extract_cmvn_from_json(path):
    with open(path, 'r', encoding='utf8') as fin:
        cmvn_stats = json.load(fin)
    sum = cmvn_stats['mean_stat']
    squere_sum = cmvn_stats['var_stat']
    count = cmvn_stats['frame_num']

    sum_np = np.array(sum)
    squere_sum_np = np.array(squere_sum)

    mean = sum_np / count
    mean.astype(np.float32)
    std = np.sqrt(np.maximum(squere_sum_np / count - mean ** 2, 1e-20))
    std.astype(np.float32)

    if DEBUG:
        pdb.set_trace()

    np.save('{}.mean.npy'.format(path), mean)
    np.save('{}.std.npy'.format(path), std)
    return mean, std

def combine_cmvn_stats(output_path, *paths, ratio=None):
    sum_np = None
    squere_sum_np = None
    count = 0
    first_num = None
    for path in paths:
        with open(path, 'r', encoding='utf8') as fin:
            cmvn_stats = json.load(fin)
        temp_sum_np = np.array(cmvn_stats['mean_stat'])
        temp_squere_sum_np = np.array(cmvn_stats['var_stat'])
        temp_count = cmvn_stats['frame_num']
        if first_num is None:
            first_num = count
        elif ratio is not None:
            assert ratio > 0
            sum_np_dtype = temp_sum_np.dtype
            squere_sum_np_dtype = temp_squere_sum_np.dtype
            zoom_factor = first_num / ratio / temp_count
            temp_sum_np *= zoom_factor
            temp_sum_np.astype(sum_np_dtype)
            temp_squere_sum_np *= zoom_factor
            temp_squere_sum_np.astype(squere_sum_np_dtype)
            temp_count = int(temp_count * zoom_factor)
        if sum_np is None:
            sum_np = np.zeros_like(temp_sum_np)
            squere_sum_np = np.zeros_like(temp_squere_sum_np)
        else:
            assert sum_np.shape == temp_sum_np.shape
            assert squere_sum_np.shape == temp_squere_sum_np.shape
        sum_np += temp_sum_np
        squere_sum_np += temp_squere_sum_np
        count += temp_count
    with open(output_path, 'w', encoding='utf8') as fout:
        json.dump({
            'mean_stat': sum_np.tolist(),
            'var_stat': squere_sum_np.tolist(),
            'frame_num': count
        }, fout)


if __name__ == '__main__':
    aishell_path = 'dump/corpus/aishell/train/global_cmvn'
    aishell_80_path = 'dump/corpus/aishell/train/global_cmvn_80'
    wenetspeech_path = 'egs/wenetspeech/data/train/global_cmvn'
    wenetspeech_80_path = 'egs/wenetspeech/data/train/global_cmvn_80_segments_50000'
    ws_segments_50000_path = 'egs/wenetspeech/data/train/global_cmvn_segments_50000'
    combine_path = 'dump/corpus/aishell_wenetspeech/global_cmvn'

    aishell_mean, aishell_std = extract_cmvn_from_json(aishell_80_path)
    ws_mean, ws_std = extract_cmvn_from_json(wenetspeech_80_path)
    # ws50000_mean, ws50000_std = extract_cmvn_from_json(ws_segments_50000_path)
    # combine_mean, combine_std = extract_cmvn_from_json(combine_path)

    # combine_cmvn_stats(combine_path, aishell_path, wenetspeech_path)


    if DEBUG:
        pdb.set_trace()