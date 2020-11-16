#!/usr/bin/env bash
chmod +x /userhome/code/CAP/prep_imagenet.sh
cd /userhome/code/CAP
echo "preparing data"
bash /userhome/code/CAP/prep_imagenet.sh >> /dev/null
echo "preparing data finished"
python3 -m torch.distributed.launch --nproc_per_node=4 train.py --train \
 --model "mobilenetv2" \
 --cfg "[28, 14, 18, 26, 62, 62, 158, 264]" \
 --path "Exp_train/train_mobilenetv2_220m_${RANDOM}" \
 --dataset "imagenet" \
 --save_path "/dev/shm/imagenet" \
 --base_path "Exp_base/mobilenetv2_base" \
 --warm_epoch 1 \
 --sync_bn \
 --n_epochs 120 \
 --label_smoothing 0
