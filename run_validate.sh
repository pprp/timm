#!/bin/bash
DATA=/home/inspur/data/imagenet

CUDA_VISIBLE_DEVICES=0 python validate.py $DATA --model mp_mobilenet_v2 --num-gpu=1 --checkpoint /home/inspur/project/timm/metapooling_with_procedureB/20221012-195709-mp_mobilenet_v2-224/checkpoint-288.pth.tar --num-classes 1000 --img-size 224

# `python validate.py /imagenet/validation/ --model seresnext26_32x4d --pretrained`
# --test-pool 