#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 evaluate.py \
 --model "resnet34" \
 --path "Exp_search/search_resnet34_2000m" \
 --dataset "imagenet" \
 --save_path "/userhome/data/imagenet" \
 --cfg "[54, 30, 48, 22, 30, 96, 68, 80, 86, 114, 180, 248, 134, 244, 120, 222, 162, 348, 512, 414, 256]"