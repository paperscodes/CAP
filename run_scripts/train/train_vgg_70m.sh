#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=2 train.py --train \
 --model "vgg" \
 --cfg "[22, 34, 58, 72, 152, 126, 118, 270, 198, 202, 260, 102, 216]" \
 --path "Exp_train/train_vgg_70m" \
 --dataset "cifar10" \
 --save_path "/userhome/data/cifar10" \
 --base_path "Exp_base/vgg_base" \
 --warm_epoch 1 \
 --n_epochs 300 \
 --sync_bn