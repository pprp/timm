#!/bin/bash
NUM_PROC=$1
PORT=$2
shift
shift

python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port $PORT train.py "$@"