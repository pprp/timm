#!/bin/bash
# DATA=/home/inspur/data/imagenet

module load cuda/11.3
module load anaconda/2021.05
source activate torchgpu
DATA=/data/public/imagenet2012

# random training  
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./distributed_train.sh 4 41291 $DATA \
    --model rand_mp_mobilenet_v2 \
    -b 128 -j 4 \
    --sched cosine --epochs 200 \
    --decay-epochs 2.4 --decay-rate .973 \
    --opt lamb --opt-eps 1e-06 \
    --warmup-lr 1e-6 --weight-decay 0.09 --warmup-epochs 5 \
    --drop 0.1 --drop-path 0.08 \
    --aa rand-m9-mstd0.5 --remode pixel \
    --reprob 0.2 --amp \
    --lr 0.0025 --lr-noise 0.42 0.9 \
    --interpolation bicubic --min-lr 1e-5 --mixup 0.15 \
    --num-classes 1000 \
    --output random_ablation_study \
    --seed 42 \
    --train-interpolation bicubic

# # global training 
# CUDA_VISIBLE_DEVICES=0 bash ./distributed_train.sh 1 44291 $DATA \
#     --model global_mp_mobilenet_v2 \
#     -b 512 -j 2 \
#     --sched cosine --epochs 200 \
#     --decay-epochs 2.4 --decay-rate .973 \
#     --opt lamb --opt-eps 1e-06 \
#     --warmup-lr 1e-6 --weight-decay 0.09 --warmup-epochs 5 \
#     --drop 0.1 --drop-path 0.08 \
#     --aa rand-m9-mstd0.5 --remode pixel \
#     --reprob 0.2 --amp \
#     --lr 0.005 --lr-noise 0.42 0.9 \
#     --interpolation bicubic --min-lr 1e-5 --mixup 0.15 \
#     --num-classes 1000 \
#     --output global_ablation_study \
#     --seed 42 \
#     --train-interpolation bicubic

# # local training 
# CUDA_VISIBLE_DEVICES=0 bash ./distributed_train.sh 1 44191 $DATA \
#     --model local_mp_mobilenet_v2 \
#     -b 512 -j 2 \
#     --sched cosine --epochs 200 \
#     --decay-epochs 2.4 --decay-rate .973 \
#     --opt lamb --opt-eps 1e-06 \
#     --warmup-lr 1e-6 --weight-decay 0.09 --warmup-epochs 5 \
#     --drop 0.1 --drop-path 0.08 \
#     --aa rand-m9-mstd0.5 --remode pixel \
#     --reprob 0.2 --amp \
#     --lr 0.005 --lr-noise 0.42 0.9 \
#     --interpolation bicubic --min-lr 1e-5 --mixup 0.15 \
#     --num-classes 1000 \
#     --output local_ablation_study \
#     --seed 42 \
#     --train-interpolation bicubic
