#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 evaluate.py \
 --model "resnet18" \
 --path "Exp_search/search_resnet18_1000m" \
 --dataset "imagenet" \
 --save_path "/userhome/data/imagenet" \
 --cfg "[52, 26, 54, 44, 82, 104, 110, 154, 164, 100, 318, 502, 418]"