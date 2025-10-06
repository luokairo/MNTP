#!/bin/bash

# 设置训练环境变量
export CUDA_VISIBLE_DEVICES=0  # 设置使用的GPU

# 单节点训练
torchrun --nproc_per_node=1 --master_port=29502 /fs/scratch/PAS2473/MM2025/CVPR2026/MNTP/train.py \
  --depth=16 \
  --bs=468 \
  --ep=1 \
  --fp16=1 \
  --tblr=1e-4 \
  --alng=1e-3 \
  --wpe=0.1 \
  --data_load_reso=256 \