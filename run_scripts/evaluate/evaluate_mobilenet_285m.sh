#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 evaluate.py \
 --model "mobilenet" \
 --path "Exp_search/search_mobilenet_285m" \
 --dataset "imagenet" \
 --save_path "/userhome/data/imagenet" \
 --cfg "[30, 62, 74, 52, 216, 210, 478, 222, 436, 332, 206, 444, 646, 1002]"