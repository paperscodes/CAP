#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 evaluate.py \
 --model "resnet50" \
 --path "Exp_search/search_resnet50_1800m" \
 --dataset "imagenet" \
 --save_path "/userhome/data/imagenet" \
 --cfg "[44, 56, 28, 214, 34, 46, 42, 58, 94, 76, 322, 76, 64, 84, 58, 30, 30, 186, 188, 670, 192, 126, 128, 176, 136, 146, 122, 182, 140, 228, 280, 364, 1824, 268, 262, 260, 490]"