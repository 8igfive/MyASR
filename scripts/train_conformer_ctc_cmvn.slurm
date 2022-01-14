#!/bin/bash
#SBATCH -J train_conformer_ctc_cmvn
#SBATCH -o egs/wenetspeech/log/%j.log
#SBATCH -n 8
#SBATCH --cpus-per-task=4
#SBATCH -p sugon
#SBATCH -w sugon-gpu-3
#SBATCH --gres=gpu:8

echo 'train conformer encoder with ctc and cmvn'
nvidia-smi
/home/LAB/qujy/.conda/envs/wenet/bin/python run_lmx.py -c scripts/conf/train_conformer_ctc_cmvn.yaml -ct -im egs/wenetspeech/exp/Conformer_CTC_CMVN/model.now.pt -tfe 13 -tfs 177000 -n 8 -g 0,1,2,3,4,5,6,7 -p ddp -r 0 &
/home/LAB/qujy/.conda/envs/wenet/bin/python run_lmx.py -c scripts/conf/train_conformer_ctc_cmvn.yaml -ct -im egs/wenetspeech/exp/Conformer_CTC_CMVN/model.now.pt -tfe 13 -tfs 177000 -n 8 -g 0,1,2,3,4,5,6,7 -p ddp -r 1 &
/home/LAB/qujy/.conda/envs/wenet/bin/python run_lmx.py -c scripts/conf/train_conformer_ctc_cmvn.yaml -ct -im egs/wenetspeech/exp/Conformer_CTC_CMVN/model.now.pt -tfe 13 -tfs 177000 -n 8 -g 0,1,2,3,4,5,6,7 -p ddp -r 2 &
/home/LAB/qujy/.conda/envs/wenet/bin/python run_lmx.py -c scripts/conf/train_conformer_ctc_cmvn.yaml -ct -im egs/wenetspeech/exp/Conformer_CTC_CMVN/model.now.pt -tfe 13 -tfs 177000 -n 8 -g 0,1,2,3,4,5,6,7 -p ddp -r 3 &
/home/LAB/qujy/.conda/envs/wenet/bin/python run_lmx.py -c scripts/conf/train_conformer_ctc_cmvn.yaml -ct -im egs/wenetspeech/exp/Conformer_CTC_CMVN/model.now.pt -tfe 13 -tfs 177000 -n 8 -g 0,1,2,3,4,5,6,7 -p ddp -r 4 &
/home/LAB/qujy/.conda/envs/wenet/bin/python run_lmx.py -c scripts/conf/train_conformer_ctc_cmvn.yaml -ct -im egs/wenetspeech/exp/Conformer_CTC_CMVN/model.now.pt -tfe 13 -tfs 177000 -n 8 -g 0,1,2,3,4,5,6,7 -p ddp -r 5 &
/home/LAB/qujy/.conda/envs/wenet/bin/python run_lmx.py -c scripts/conf/train_conformer_ctc_cmvn.yaml -ct -im egs/wenetspeech/exp/Conformer_CTC_CMVN/model.now.pt -tfe 13 -tfs 177000 -n 8 -g 0,1,2,3,4,5,6,7 -p ddp -r 6 &
/home/LAB/qujy/.conda/envs/wenet/bin/python run_lmx.py -c scripts/conf/train_conformer_ctc_cmvn.yaml -ct -im egs/wenetspeech/exp/Conformer_CTC_CMVN/model.now.pt -tfe 13 -tfs 177000 -n 8 -g 0,1,2,3,4,5,6,7 -p ddp -r 7 &
wait