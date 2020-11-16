#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 evaluate.py \
 --model "vgg" \
 --path "Exp_search/search_vgg_150m" \
 --dataset "cifar10" \
 --save_path "/userhome/data/cifar10" \
 --cfg "[64, 46, 82, 94, 174, 182, 172, 354, 204, 376, 266, 512, 460]"