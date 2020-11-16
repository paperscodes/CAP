#!/usr/bin/env bash
chmod +x /userhome/code/CAP/prep_imagenet.sh
cd /userhome/code/CAP
echo "preparing data"
bash /userhome/code/CAP/prep_imagenet.sh >> /dev/null
echo "preparing data finished"
python3 -m torch.distributed.launch --nproc_per_node=4 train.py --train \
 --model "mobilenet" \
 --cfg "[14, 32, 86, 20, 116, 86, 332, 160, 186, 86, 106, 282, 464, 790]" \
 --path "Exp_train/train_mobilenet_100m_${RANDOM}" \
 --dataset "imagenet" \
 --save_path "/dev/shm/imagenet" \
 --base_path "Exp_base/mobilenet_base" \
 --warm_epoch 1 \
 --sync_bn \
 --n_epochs 120
