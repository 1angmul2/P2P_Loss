#!/usr/bin/env bash

PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES="0,2,3,5" \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=$PORT \
    $(dirname "$0")/train.py --launcher pytorch ${@:3}