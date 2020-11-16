#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=2 train.py --train \
 --model "vgg" \
 --cfg "[20, 34, 94, 90, 174, 128, 186, 338, 190, 286, 150, 198, 266]" \
 --path "Exp_train/train_vgg_100m" \
 --dataset "cifar10" \
 --save_path "/userhome/data/cifar10" \
 --base_path "Exp_base/vgg_base" \
 --warm_epoch 1 \
 --n_epochs 300 \
 --sync_bn