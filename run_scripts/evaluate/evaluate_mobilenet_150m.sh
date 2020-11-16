#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 evaluate.py \
 --model "mobilenet" \
 --path "Exp_search/search_mobilenet_150m" \
 --dataset "imagenet" \
 --save_path "/userhome/data/imagenet" \
 --cfg "[18, 26, 110, 38, 132, 90, 406, 194, 258, 248, 196, 288, 502, 774]"