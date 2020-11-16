#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 evaluate.py \
 --model "resnet50" \
 --path "Exp_search/search_resnet50_600m" \
 --dataset "imagenet" \
 --save_path "/userhome/data/imagenet" \
 --cfg "[24, 32, 28, 96, 20, 30, 32, 22, 24, 30, 132, 30, 30, 50, 28, 40, 24, 134, 138, 288, 92, 46, 74, 48, 104, 90, 66, 148, 98, 152, 172, 306, 1168, 190, 228, 164, 260]"