#!/bin/bash
export OMP_NUM_THREADS=40
NUM_PROC=$1
shift
/home/qinch21/miniconda3/envs/torch/bin/python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC train.py "$@"

