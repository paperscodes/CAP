#!/usr/bin/env bash
chmod +x /userhome/code/CAP/prep_imagenet.sh
cd /userhome/code/CAP
echo "preparing data"
bash /userhome/code/CAP/prep_imagenet.sh >> /dev/null
echo "preparing data finished"
python3 -m torch.distributed.launch --nproc_per_node=4 train.py --train \
 --model "resnet50" \
 --cfg "[24, 32, 28, 96, 20, 30, 32, 22, 24, 30, 132, 30, 30, 50, 28, 40, 24, 134, 138, 288, 92, 46, 74, 48, 104, 90, 66, 148, 98, 152, 172, 306, 1168, 190, 228, 164, 260]" \
 --path "Exp_train/train_resnet50_600m_${RANDOM}" \
 --dataset "imagenet" \
 --save_path "/dev/shm/imagenet" \
 --base_path "Exp_base/resnet50_base" \
 --sync_bn \
 --warm_epoch 1 \
 --n_epochs 120
