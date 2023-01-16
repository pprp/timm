#!/bin/bash

module load cuda/11.3
module load anaconda/2021.05
source activate torchgpu

DATA=/data/public/imagenet2012

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./distributed_train.sh 8 44331 $DATA \
    --model mp_mobilenet_v2_075 \
    -b 256 -j 8 \
    --sched cosine --epochs 300 \
    --decay-epochs 2.4 --decay-rate .973 \
    --opt lamb --opt-eps 1e-06 \
    --warmup-lr 1e-6 --weight-decay 0.1 --warmup-epochs 5 \
    --drop 0.1 --drop-path 0.08 \
    --aa rand-m9-mstd0.5 --remode pixel \
    --reprob 0.2 --amp \
    --lr 0.005 --lr-noise 0.42 0.9 \
    --interpolation bicubic --min-lr 1e-5 --mixup 0.15 \
    --num-classes 1000 \
    --output metapooling_with_procedureB \
    --seed 666 \
    --train-interpolation bicubic
