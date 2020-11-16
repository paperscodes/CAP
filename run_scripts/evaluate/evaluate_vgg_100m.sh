#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 evaluate.py \
 --model "vgg" \
 --path "Exp_search/search_vgg_100m" \
 --dataset "cifar10" \
 --save_path "/userhome/data/cifar10" \
 --cfg "[20, 34, 94, 90, 174, 128, 186, 338, 190, 286, 150, 198, 266]"