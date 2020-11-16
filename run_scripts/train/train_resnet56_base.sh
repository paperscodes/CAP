#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=2 train.py --train \
 --model "resnet56" \
 --path "Exp_base/resnet56_base_${RANDOM}" \
 --dataset "cifar10" \
 --save_path "/userhome/data/cifar10" \
 --warm_epoch 5 \
 --n_epochs 300 \
 --sync_bn