#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 evaluate.py \
 --model "resnet50" \
 --path "Exp_search/search_resnet50_1000m" \
 --dataset "imagenet" \
 --save_path "/userhome/data/imagenet" \
 --cfg "[40, 28, 20, 72, 36, 32, 36, 40, 28, 74, 166, 54, 74, 92, 40, 56, 32, 140, 162, 550, 118, 80, 106, 96, 158, 174, 66, 90, 68, 118, 188, 304, 1324, 312, 344, 340, 334]"