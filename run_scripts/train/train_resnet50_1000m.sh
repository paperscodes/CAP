#!/usr/bin/env bash
chmod +x /userhome/code/CAP/prep_imagenet.sh
cd /userhome/code/CAP
echo "preparing data"
bash /userhome/code/CAP/prep_imagenet.sh >> /dev/null
echo "preparing data finished"
python3 -m torch.distributed.launch --nproc_per_node=4 train.py --train \
 --model "resnet50" \
 --cfg "[40, 28, 20, 72, 36, 32, 36, 40, 28, 74, 166, 54, 74, 92, 40, 56, 32, 140, 162, 550, 118, 80, 106, 96, 158, 174, 66, 90, 68, 118, 188, 304, 1324, 312, 344, 340, 334]" \
 --path "Exp_train/train_resnet50_1000m_${RANDOM}" \
 --dataset "imagenet" \
 --save_path "/dev/shm/imagenet" \
 --base_path "Exp_base/resnet50_base" \
 --sync_bn \
 --warm_epoch 1 \
 --n_epochs 120
