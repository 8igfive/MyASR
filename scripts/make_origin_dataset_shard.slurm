#!/bin/bash
#SBATCH -J make_origin_dataset_shard
#SBATCH -o scripts/log/%j.log
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p cpu

/home/LAB/qujy/.conda/envs/wenet/bin/python tools/make_shard_list.py --remove_blanks --num_utts_per_shard 1000 \
                                                                     --num_threads 16 \
                                                                     --resample 16000 \
                                                                     egs/aishell/data/train/wav.scp \
                                                                     egs/aishell/data/train/text \
                                                                     dump/corpus/aishell/test \
                                                                     dump/corpus/aishell/test/data86.list &
# /home/LAB/qujy/.conda/envs/wenet/bin/python tools/make_cvcorpus_list.py &
wait