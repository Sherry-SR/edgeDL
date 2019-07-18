#!/usr/bin/env bash

mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
srun -p MIA -n1 -w SH-IDC1-10-5-38-150 --gres=gpu:1 --mpi=pmi2 --job-name=pelvis --kill-on-bad-exit=1 python /mnt/lustre/shenrui/project/edgeDL/train.py --config /mnt/lustre/shenrui/project/edgeDL/resources/train_config_ce.yaml >log/train-$now.log 2>&1 &
