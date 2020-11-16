#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 evaluate.py \
 --model "mobilenetv2" \
 --path "Exp_search/search_mobilenetv2_220m" \
 --dataset "imagenet" \
 --save_path "/userhome/data/imagenet" \
 --cfg "[28, 14, 18, 26, 62, 62, 158, 264]"