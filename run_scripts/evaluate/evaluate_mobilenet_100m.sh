#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 evaluate.py \
 --model "mobilenet" \
 --path "Exp_search/search_mobilenet_100m" \
 --dataset "imagenet" \
 --save_path "/userhome/data/imagenet" \
 --cfg "[14, 32, 86, 20, 116, 86, 332, 160, 186, 86, 106, 282, 464, 790]"