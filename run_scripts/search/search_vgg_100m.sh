#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=2 search.py --train \
 --model "vgg" \
 --path "Exp_search/search_vgg_100m_${RANDOM}" \
 --dataset "cifar10" \
 --save_path "/userhome/data/cifar10" \
 --target_flops 100000000 \
 --search_epoch 10 \
 --n_remove 2 \
 --n_best 3 \
 --n_populations 8 \
 --n_generations 25 \
 --sync_bn \
 --div 2 \
 --n_epochs 300 \
 --manual_seed ${RANDOM} \
 --label_smoothing 0
