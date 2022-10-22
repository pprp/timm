#!/bin/bash
DATA=/home/inspur/data/imagenet

# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./distributed_train.sh 4 44331 $DATA \
#     --model mp_mobilenet_v2 \
#     -b 512 -j 4 \
#     --sched cosine --epochs 320 \
#     --decay-epochs 2.4 --decay-rate .973 \
#     --opt lamb --opt-eps 1e-06 \
#     --warmup-lr 1e-6 --weight-decay 0.1 --warmup-epochs 5 \
#     --drop 0.1 --drop-path 0.08 \
#     --aa rand-m9-mstd0.5 --remode pixel \
#     --reprob 0.2 --amp \
#     --lr 0.005 --lr-noise 0.42 0.9 \
#     --interpolation bicubic --min-lr 1e-5 --mixup 0.15 \
#     --num-classes 1000 \
#     --output metapooling_with_procedureB \
#     --resume /home/inspur/project/timm/metapooling_with_procedureB/20221012-195709-mp_mobilenet_v2-224/checkpoint-304.pth.tar \
#     --seed 666 \
#     --train-interpolation bicubic

CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./distributed_train.sh 4 47331 $DATA \
    --model mp_mobilenet_v2_075 \
    -b 512 -j 4 \
    --sched cosine --epochs 240 \
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
