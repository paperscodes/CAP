#!/usr/bin/env bash
chmod +x /userhome/code/CAP/prep_imagenet.sh
cd /userhome/code/CAP
echo "preparing data"
bash /userhome/code/CAP/prep_imagenet.sh >> /dev/null
echo "preparing data finished"
python3 -m torch.distributed.launch --nproc_per_node=4 train.py --train \
 --model "mobilenetv2" \
 --cfg "[20, 8, 18, 18, 50, 50, 138, 196]" \
 --path "Exp_train/train_mobilenetv2_150m_${RANDOM}" \
 --dataset "imagenet" \
 --save_path "/dev/shm/imagenet" \
 --base_path "Exp_base/mobilenetv2_base" \
 --warm_epoch 1 \
 --sync_bn \
 --n_epochs 120 \
 --label_smoothing 0
