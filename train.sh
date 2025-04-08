#!/bin/bash

CUDA_VISIBLE_DEVICES=2 torchrun \
    --nproc_per_node=1 \
    --master_port 12340 \
    train/train.py \
    --config ./config/base.yaml \
