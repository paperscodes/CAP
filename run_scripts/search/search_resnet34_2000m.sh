#!/usr/bin/env bash
chmod +x /userhome/code/CAP/prep_imagenet.sh
cd /userhome/code/CAP
echo "preparing data"
bash /userhome/code/CAP/prep_imagenet.sh >> /dev/null
echo "preparing data finished"
python3 -m torch.distributed.launch --nproc_per_node=4 search.py --train \
 --model "resnet34" \
 --path "Exp_search/search_resnet34_2000m_${RANDOM}" \
 --dataset "imagenet" \
 --save_path "/dev/shm/imagenet" \
 --target_flops 2000000000 \
 --search_epoch 3 \
 --n_remove 2 \
 --n_best 3 \
 --n_populations 8 \
 --n_generations 20 \
 --sync_bn \
 --div 2 \
 --warm_epoch 1 \
 --n_epochs 120 \
 --label_smoothing 0 \
 --manual_seed ${RANDOM}