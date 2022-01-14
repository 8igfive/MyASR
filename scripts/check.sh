#!/usr/bin/env bash
# srun -p sugon -N 1 --gres=gpu:1 /home/LAB/qujy/.conda/envs/open1/bin/python3.6 eval_lmx.py -m egs/aishell2/exp/simple_transformer/model.now.pt -c /home/LAB/qujy/open/OpenTransformer-master/egs/aishell2/conf/test_lmx.yaml -s test_0
# srun -p sugon -N 1 --gres=gpu:1 /home/LAB/qujy/.conda/envs/open1/bin/python3.6 eval_lmx.py -m egs/aishell2/exp/simple_transformer/model.now.pt -lm egs/aishell2/exp/transformer_lm/model.now.pt -c /home/LAB/qujy/open/OpenTransformer-master/egs/aishell2/conf/test_lmx.yaml -s test_0
# srun -p sugon -N 1 --gres=gpu:1 /home/LAB/qujy/.conda/envs/open1/bin/python3.6 eval_lmx.py -m egs/aishell2/exp/simple_transformer/model.now.pt -lm dump/bertpretrain_lm/bert-base-chinese_lm.ckpt -c /home/LAB/qujy/open/OpenTransformer-master/egs/aishell2/conf/test_lmx.yaml -s test_0
# srun -p sugon -n 1 --gres=gpu:1 /home/LAB/qujy/.conda/envs/wenet/bin/python eval_lmx.py -m egs/aishell2/exp/simple_transformer/model.now.pt -lm dump/bertpretrain_lm/bert-base-chinese_lm.ckpt -c /home/LAB/qujy/open/OpenTransformer-master/egs/aishell2/conf/test_lmx.yaml -s test_1

# srun -p inspur -w inspur-gpu-06 /home/LAB/qujy/.conda/envs/wenet/bin/python eval_lmx.py -m egs/wenetspeech/exp/TransWithWS/model.now.pt -lm dump/bertpretrain_lm/bert-base-chinese_lm.ckpt -c /home/LAB/qujy/open/OpenTransformer-master/egs/aishell2/conf/test_lmx.yaml -s test_1
# srun -p sugon -n 1 --gres=gpu:1 /home/LAB/qujy/.conda/envs/wenet/bin/python eval_lmx.py -m egs/aishell2/exp/simple_transformer/model.now.pt -lm dump/bertpretrain_lm/bert-base-chinese_lm.ckpt -c /home/LAB/qujy/open/OpenTransformer-master/scripts/conf/continue_with_ws.yaml -s test_1

# srun -p sugon -n 1 --gres=gpu:1 /home/LAB/qujy/.conda/envs/wenet/bin/python eval_lmx.py -m egs/aishell2/exp/TestModel_li/model.now.pt -lm dump/bertpretrain_lm/bert-base-chinese_lm.ckpt -c egs/aishell2/conf/test_lmx.yaml -s test_2
# TestModel_li
# srun -p inspur -w inspur-gpu-06 /home/LAB/qujy/.conda/envs/wenet/bin/python eval_lmx.py -m egs/wenetspeech/exp/TestModel_li/model.epoch.35.pt -lm dump/bertpretrain_lm/bert-base-chinese_lm.ckpt -c egs/aishell2/conf/test_lmx.yaml -s test_2
# srun -p inspur -w inspur-gpu-07 /home/LAB/qujy/.conda/envs/wenet/bin/python eval_lmx.py -m egs/wenetspeech/exp/TestModel_li/model.epoch.64.pt -lm dump/ernie_lm/ernie-1.0.ckpt -c scripts/conf/test_transformer.yaml # -s test_0
# TranWithAll
srun -p inspur -w inspur-gpu-07 /home/LAB/qujy/.conda/envs/wenet/bin/python eval_lmx.py -m egs/wenetspeech/exp/TranWithAll/model.epoch.8.pt -lm dump/ernie_lm/ernie-1.0.ckpt -c scripts/conf/train_with_all.yaml # -s test_1
# srun -p inspur -w inspur-gpu-07 /home/LAB/qujy/.conda/envs/wenet/bin/python eval_lmx.py -m egs/wenetspeech/exp/TranWithAll/model.epoch.8.pt -lm dump/bertpretrain_lm/bert-base-chinese_lm.ckpt -c egs/aishell2/conf/test_lmx.yaml # -s test_1
# Conformer
# srun -p inspur -w inspur-gpu-07 /home/LAB/qujy/.conda/envs/wenet/bin/python eval_lmx.py -m egs/wenetspeech/exp/Conformer_CTC_CMVN/model.epoch.9.pt -lm dump/ernie_lm/ernie-1.0.ckpt -c scripts/conf/train_conformer_ctc_cmvn.yaml # -s test_1
# srun -p inspur -w inspur-gpu-06 /home/LAB/qujy/.conda/envs/wenet/bin/python eval_lmx.py -m egs/wenetspeech/exp/Conformer_CTC_CMVN/model.now.pt -lm dump/roberta_lm/roberta_lm.ckpt  -c scripts/conf/conformer_ctc_cmvn_audio.yaml # -s test_1

# srun -p sugon -n 1 --gres=gpu:1 /home/LAB/qujy/.conda/envs/wenet/bin/python test_dataset.py
# srun -p sugon --gres=gpu:1 /home/LAB/qujy/.conda/envs/open1/bin/python3.6 test.py
# srun -p cpu /home/LAB/qujy/.conda/envs/open1/bin/python3.6 tools/noise_reduction.py -f egs/aishell/data/train/wav.scp -t egs/aishell/data/train/text
# srun -p cpu /home/LAB/qujy/.conda/envs/open1/bin/python3.6 build_bertpretrain_lm.py