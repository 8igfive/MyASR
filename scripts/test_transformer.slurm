#!/bin/bash
#SBATCH -J test_transformer
#SBATCH -o egs/wenetspeech/log/%j.log
#SBATCH -n 2
#SBATCH --cpus-per-task=4
#SBATCH -p inspur
#SBATCH -w inspur-gpu-09

echo 'test transformer'
nvidia-smi
/home/LAB/qujy/.conda/envs/wenet/bin/python run_lmx.py -ct -im egs/aishell2/exp/TestModel_li/model.now.pt -c scripts/conf/test_transformer.yaml -n 2 -g 1,2 -p ddp -r 0 -se 3 &
/home/LAB/qujy/.conda/envs/wenet/bin/python run_lmx.py -ct -im egs/aishell2/exp/TestModel_li/model.now.pt -c scripts/conf/test_transformer.yaml -n 2 -g 1,2 -p ddp -r 1 -se 3 &
# /home/LAB/qujy/.conda/envs/wenet/bin/python run_lmx.py -ct -im egs/aishell2/exp/TestModel_li/model.now.pt -c scripts/conf/test_transformer.yaml -n 4 -g 0,1,2,3 -p ddp -r 2 -se 3 &
# /home/LAB/qujy/.conda/envs/wenet/bin/python run_lmx.py -ct -im egs/aishell2/exp/TestModel_li/model.now.pt -c scripts/conf/test_transformer.yaml -n 4 -g 0,1,2,3 -p ddp -r 3 -se 3 &
wait