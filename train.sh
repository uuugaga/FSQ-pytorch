#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,6,7 torchrun \
    --nproc_per_node=4 \
    --master_port 12340 \
    train/train.py \
    --config ./config/base.yaml \
