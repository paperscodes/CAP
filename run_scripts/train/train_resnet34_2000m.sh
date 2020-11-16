#!/usr/bin/env bash
chmod +x /userhome/code/CAP/prep_imagenet.sh
cd /userhome/code/CAP
echo "preparing data"
bash /userhome/code/CAP/prep_imagenet.sh >> /dev/null
echo "preparing data finished"
python3 -m torch.distributed.launch --nproc_per_node=4 train.py --train \
 --model "resnet34" \
 --cfg "[54, 30, 48, 22, 30, 96, 68, 80, 86, 114, 180, 248, 134, 244, 120, 222, 162, 348, 512, 414, 256]" \
 --path "Exp_train/train_resnet34_2000m_${RANDOM}" \
 --dataset "imagenet" \
 --save_path "/dev/shm/imagenet" \
 --base_path "Exp_base/resnet34_base" \
 --sync_bn \
 --warm_epoch 1 \
 --n_epochs 120
