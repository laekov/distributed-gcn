#!/bin/bash
export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_RANK
source $HOME/scripts/conda.sh
python3 -u train.py 2>/dev/null
