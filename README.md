# CAP: Compact Search for Automatic Filter Pruning

Codes for our paper "CAP: Compcat Search for Automatic Filter Pruning", which efficiently search for the best pruned model in a compact search space using sparsity scaling. 

## Prerequisites

- PyTorch 1.7+
- [Apex](https://github.com/NVIDIA/apex) (for distributed running)
- [DALI](https://github.com/NVIDIA/DALI) (for accelerating data processing)
- [torchprofile](https://github.com/zhijian-liu/torchprofile) (for FLOPs calculation, which is based on `torch.jit.trace`, more general and accurate)

## Usage

Please firstly download the pretrained models, which include [Exp_base](https://drive.google.com/drive/folders/1WH2FcqujbtLprf1XdV-Evo_aw3ODte4Y?usp=sharing) and [Exp_search](https://drive.google.com/drive/folders/1G5RPRiUDOdYVuVLW9hSQA2t5OVIIFqMA?usp=sharing)

Then you can easily run our codes with the following scripts

**Evaluation**

```bash
bash ./run_scripts/evaluate/evaluate_mobilenet_50m.sh
```

or

```python
python3 -m torch.distributed.launch --nproc_per_node=1 evaluate.py \
 --model "mobilenet" \
 --path "Exp_search/search_mobilenet_50m" \
 --dataset "imagenet" \
 --save_path "/userhome/data/imagenet" \
 --cfg "[6, 18, 42, 24, 74, 74, 196, 104, 106, 46, 148, 170, 320, 660]"
```

**Train**

```bash
bash ./run_scripts/train/train_mobilenet_50m.sh
```

or

```python
# [optional]cache imagenet dataset in RAM for accelerting I/O
chmod +x /userhome/code/CAP/prep_imagenet.sh
cd /userhome/code/CAP
echo "preparing data"
bash /userhome/code/CAP/prep_imagenet.sh >> /dev/null
echo "preparing data finished"

# distributed running, nproc_per_node=num_of_gpus for one machine multi-gpus
python3 -m torch.distributed.launch --nproc_per_node=4 train.py --train \
 --model "mobilenet" \
 --cfg "[6, 18, 42, 24, 74, 74, 196, 104, 106, 46, 148, 170, 320, 660]" \
 --path "Exp_train/train_mobilenet_50m_${RANDOM}" \
 --dataset "imagenet" \
 --save_path "/dev/shm/imagenet" \
 --base_path "Exp_base/mobilenet_base" \
 --warm_epoch 1 \
 --sync_bn \
 --n_epochs 120
```

**Search**

```bash
bash ./run_scripts/train/search_mobilenet_50m.sh
```

or

```python
# [optional]cache imagenet dataset in RAM for accelerting I/O
chmod +x /userhome/code/CAP/prep_imagenet.sh
cd /userhome/code/CAP
echo "preparing data"
bash /userhome/code/CAP/prep_imagenet.sh >> /dev/null
echo "preparing data finished"

# distributed running, nproc_per_node=num_of_gpus for one machine multi-gpus
python3 -m torch.distributed.launch --nproc_per_node=4 search.py --train \
 --model "mobilenet" \
 --path "Exp_search/search_mobilenet_50m_${RANDOM}" \
 --dataset "imagenet" \
 --save_path "/dev/shm/imagenet" \
 --target_flops 50000000 \
 --search_epoch 3 \
 --n_remove 2 \
 --n_best 3 \
 --n_populations 8 \
 --n_generations 20 \
 --div 2 \
 --warm_epoch 1 \
 --sync_bn \
 --n_epochs 120 \
 --label_smoothing 0 \
 --manual_seed ${RANDOM}

```



