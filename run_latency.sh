#!/bin/bash 
module load cuda/11.4
module load anaconda/2020.11
source activate py38

python latency_measure.py