#!/bin/bash
#SBATCH -J test_model
#SBATCH -o scripts/log/%j.log
#SBATCH -n 3
#SBATCH --cpus-per-task=4
#SBATCH -p inspur 
#SBATCH -w inspur-gpu-08

/home/LAB/qujy/.conda/envs/wenet/bin/python eval_lmx.py -m egs/wenetspeech/exp/TranWithAll/model.epoch.8.pt -lm dump/bertpretrain_lm/bert-base-chinese_lm.ckpt -c scripts/conf/test_transformer.yaml -s ws_dev
/home/LAB/qujy/.conda/envs/wenet/bin/python eval_lmx.py -m egs/wenetspeech/exp/TranWithAll/model.epoch.8.pt -lm dump/bertpretrain_lm/bert-base-chinese_lm.ckpt -c egs/aishell2/conf/test_lmx.yaml -s aishell
