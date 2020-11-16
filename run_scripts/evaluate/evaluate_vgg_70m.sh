#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 evaluate.py \
 --model "vgg" \
 --path "Exp_search/search_vgg_70m" \
 --dataset "cifar10" \
 --save_path "/userhome/data/cifar10" \
 --cfg "[22, 34, 58, 72, 152, 126, 118, 270, 198, 202, 260, 102, 216]"