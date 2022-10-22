#!/bin/bash
DATA=/home/inspur/data/imagenet

# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./distributed_train.sh 4 $DATA \
#     --model mp_mobilenet_v2 \
#     -b 512 -j 7 \
#     --sched cosine --epochs 300 \
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
#     --seed 666 \
#     --train-interpolation bicubic

# failed
# CUDA_VISIBLE_DEVICES=4,5,6,7  bash  ./distributed_train.sh 4 13344 $DATA --model mp_mobilenet_v2 -b 512 --sched step --epochs 300 --decay-epochs 2.4 --decay-rate .973 --opt rmsproptf --opt-eps .001 -j 4 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-path 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .128 --lr-noise 0.42 0.9  --num-classes 1000

# finetune from 197
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./distributed_train.sh 8 48192 $DATA \
    --model mp_mobilenet_v2_050 \
    -b 256 -j 8 \
    --sched cosine --epochs 200 \
    --decay-epochs 2.4 --decay-rate .973 \
    --opt lamb --opt-eps 1e-06 \
    --warmup-lr 1e-6 --weight-decay 0.09 --warmup-epochs 5 \
    --drop 0.1 --drop-path 0.08 \
    --aa rand-m9-mstd0.5 --remode pixel \
    --reprob 0.2 --amp \
    --lr 0.005 --lr-noise 0.42 0.9 \
    --interpolation bicubic --min-lr 1e-5 --mixup 0.15 \
    --num-classes 1000 \
    --output metapooling_with_finetune \
    --resume /home/inspur/project/timm/metapooling_with_procedureB/best_66.9_checkpoint/best_66.9.pth.tar \
    --seed 42 \
    --train-interpolation bicubic
