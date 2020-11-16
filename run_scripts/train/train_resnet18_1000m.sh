#!/usr/bin/env bash
chmod +x /userhome/code/CAP/prep_imagenet.sh
cd /userhome/code/CAP
echo "preparing data"
bash /userhome/code/CAP/prep_imagenet.sh >> /dev/null
echo "preparing data finished"
python3 -m torch.distributed.launch --nproc_per_node=4 train.py --train \
 --model "resnet18" \
 --cfg "[52, 26, 54, 44, 82, 104, 110, 154, 164, 100, 318, 502, 418]" \
 --path "Exp_train/train_resnet18_1000m_${RANDOM}" \
 --dataset "imagenet" \
 --save_path "/dev/shm/imagenet" \
 --base_path "Exp_base/resnet18_base" \
 --warm_epoch 1 \
 --sync_bn \
 --n_epochs 120
