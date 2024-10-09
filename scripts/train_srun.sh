#!/bin/bash

#SBATCH --output=logs/%j.out
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128

srun \
    accelerate launch \
    --multi_gpu --num_processes 8 --main_process_port 10000 \
    esreal/train.py
