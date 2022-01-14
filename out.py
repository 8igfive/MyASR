import re
import os
import yaml
import time
import torch
import logging
import argparse
import editdistance
from otrans.model import End2EndModel, LanguageModel
from otrans.recognize import build_recognizer
from otrans.data.loader import FeatureLoader
from otrans.train.utils import map_to_cuda
from  tqdm import tqdm

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def main(args):

    checkpoint = torch.load(args.load_model, map_location='cuda:0')
    with open('checkpoint','w') as f:

        print(checkpoint,file=f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    parser.add_argument('-n', '--ngpu', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-nb', '--nbest', type=int, default=1)
    parser.add_argument('-bw', '--beam_width', type=int, default=5)
    parser.add_argument('-pn', '--penalty', type=float, default=0.6)
    parser.add_argument('-ld', '--lamda', type=float, default=5)
    parser.add_argument('-m', '--load_model', type=str, default=None)
    parser.add_argument('-lm', '--load_language_model', type=str, default=None)
    parser.add_argument('-ngram', '--ngram_lm', type=str, default=None)
    parser.add_argument('-alpha', '--alpha', type=float, default=0.1)
    parser.add_argument('-beta', '--beta', type=float, default=0.0)
    parser.add_argument('-lmw', '--lm_weight', type=float, default=0.1)
    parser.add_argument('-cw', '--ctc_weight', type=float, default=0.0)
    parser.add_argument('-d', '--decode_set', type=str, default='test')
    parser.add_argument('-ml', '--max_len', type=int, default=60)
    parser.add_argument('-md', '--mode', type=str, default='beam')
    # transducer related
    parser.add_argument('-mt', '--max_tokens_per_chunk', type=int, default=5)
    parser.add_argument('-pf', '--path_fusion', action='store_true', default=False)
    parser.add_argument('-s', '--suffix', type=str, default=None)
    parser.add_argument('-p2w', '--piece2word', action='store_true', default=False)
    parser.add_argument('-resc', '--apply_rescoring', action='store_true', default=False)
    parser.add_argument('-lm_resc', '--apply_lm_rescoring', action='store_true', default=False)
    parser.add_argument('-rw', '--rescore_weight', type=float, default=1.0)
    parser.add_argument('-debug', '--debug', action='store_true', default=False)
    parser.add_argument('-sba', '--sort_by_avg_score', action='store_true', default=False)
    parser.add_argument('-ns', '--num_sample', type=int, default=1)
    cmd_args = parser.parse_args()

    main(cmd_args)
