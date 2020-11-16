#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=2 train.py --train \
 --model "vgg" \
 --cfg "[64, 46, 82, 94, 174, 182, 172, 354, 204, 376, 266, 512, 460]" \
 --path "Exp_train/train_vgg_150m" \
 --dataset "cifar10" \
 --save_path "/userhome/data/cifar10" \
 --base_path "Exp_base/vgg_base" \
 --warm_epoch 1 \
 --n_epochs 300 \
 --sync_bn