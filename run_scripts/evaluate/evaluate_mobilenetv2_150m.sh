#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 evaluate.py \
 --model "mobilenetv2" \
 --path "Exp_search/search_mobilenetv2_150m" \
 --dataset "imagenet" \
 --save_path "/userhome/data/imagenet" \
 --cfg "[20, 8, 18, 18, 50, 50, 138, 196]"