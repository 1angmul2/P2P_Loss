#!/usr/bin/env bash

PORT=${PORT:-29509}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES="0,1,2,7" \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=$PORT \
    $(dirname "$0")/test.py --launcher pytorch ${@:3}