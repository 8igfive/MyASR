#!/bin/bash
#SBATCH -J test_lmx
#SBATCH -o /home/LAB/qujy/open/OpenTransformer-master/egs/aishell2/logs/%j.log
#SBATCH -n 2
#SBATCH -p inspur
#SBATCH -w inspur-gpu-06

echo 'test conformer'
nvidia-smi
/home/LAB/qujy/.conda/envs/wenet/bin/python3.6 run_lmx.py  -c /home/LAB/qujy/open/OpenTransformer-master/scripts/conf/test_conformer.yaml -n 2 -g 0 -p ddp -r 0 &
/home/LAB/qujy/.conda/envs/wenet/bin/python3.6 run_lmx.py  -c /home/LAB/qujy/open/OpenTransformer-master/scripts/conf/test_conformer.yaml -n 2 -g 1 -p ddp -r 1 &
wait