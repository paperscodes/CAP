#!/usr/bin/env bash
chmod +x /userhome/code/CAP/prep_imagenet.sh
cd /userhome/code/CAP
echo "preparing data"
bash /userhome/code/CAP/prep_imagenet.sh >> /dev/null
echo "preparing data finished"
python3 -m torch.distributed.launch --nproc_per_node=4 train.py --train \
 --model "resnet50" \
 --cfg "[44, 56, 28, 214, 34, 46, 42, 58, 94, 76, 322, 76, 64, 84, 58, 30, 30, 186, 188, 670, 192, 126, 128, 176, 136, 146, 122, 182, 140, 228, 280, 364, 1824, 268, 262, 260, 490]" \
 --path "Exp_train/train_resnet50_1800m_${RANDOM}" \
 --dataset "imagenet" \
 --save_path "/dev/shm/imagenet" \
 --base_path "Exp_base/resnet50_base" \
 --sync_bn \
 --warm_epoch 1 \
 --n_epochs 120
