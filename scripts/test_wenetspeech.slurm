#!/bin/bash
#SBATCH -J test_wenetspeech
#SBATCH -o egs/wenetspeech/log/%j.log
#SBATCH -n 2
#SBATCH --cpus-per-task=4
#SBATCH -p inspur
#SBATCH -w inspur-gpu-07

echo 'test wenetspeech'
nvidia-smi
/home/LAB/qujy/.conda/envs/wenet/bin/python run_lmx.py -ct -im egs/aishell2/exp/TestModel_li/model.now.pt -c scripts/conf/continue_with_ws.yaml -n 2 -g 2,3 -p ddp -r 0 &
/home/LAB/qujy/.conda/envs/wenet/bin/python run_lmx.py -ct -im egs/aishell2/exp/TestModel_li/model.now.pt -c scripts/conf/continue_with_ws.yaml -n 2 -g 2,3 -p ddp -r 1 &
# /home/LAB/qujy/.conda/envs/wenet/bin/python run_lmx.py -ct -im egs/wenetspeech/exp/TransWithWS/model.now.pt -c scripts/conf/continue_with_ws.yaml -n 4 -g 0,1,2,3 -p ddp -r 2 &
# /home/LAB/qujy/.conda/envs/wenet/bin/python run_lmx.py -ct -im egs/wenetspeech/exp/TransWithWS/model.now.pt -c scripts/conf/continue_with_ws.yaml -n 4 -g 0,1,2,3 -p ddp -r 3 &
wait