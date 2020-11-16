#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 evaluate.py \
 --model "mobilenet" \
 --path "Exp_search/search_mobilenet_50m" \
 --dataset "imagenet" \
 --save_path "/userhome/data/imagenet" \
 --cfg "[6, 18, 42, 24, 74, 74, 196, 104, 106, 46, 148, 170, 320, 660]"