#!/bin/bash
#SBATCH -J compute_cmvn
#SBATCH -o scripts/log/%j.log
#SBATCH -n 2
#SBATCH --cpus-per-task=16
#SBATCH -p cpu

train_config=scripts/conf/train_conformer_ctc_cmvn.yaml

# /home/LAB/qujy/.conda/envs/wenet/bin/python tools/compute_cmvn_stats.py \
#                                               --num_workers 16 \
#                                               --train_config $train_config \
#                                               --in_scp egs/wenetspeech/data/train/wav.scp.sampled_20p \
#                                               --out_cmvn egs/wenetspeech/data/train/global_cmvn_80_20p & \

/home/LAB/qujy/.conda/envs/wenet/bin/python tools/compute_cmvn_stats.py \
                                              --num_workers 16 \
                                              --train_config $train_config \
                                              --in_scp egs/aishell2/cat_wav.scp \
                                              --out_cmvn dump/corpus/aishell/train/global_cmvn_80 & \

/home/LAB/qujy/.conda/envs/wenet/bin/python tools/compute_cmvn_stats.py \
                                              --num_workers 16 \
                                              --train_config $train_config \
                                              --in_scp egs/wenetspeech/data/train/wav.scp.segments.sampled_50000 \
                                              --out_cmvn egs/wenetspeech/data/train/global_cmvn_80_segments_50000 & \

wait