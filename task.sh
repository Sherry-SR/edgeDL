#!/usr/bin/env bash

mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
srun -p MIA -n1 -w SH-IDC1-10-5-38-154 --gres=gpu:1 --mpi=pmi2 --job-name=pelvis --kill-on-bad-exit=1 python /mnt/lustre/shenrui/project/edgeDL/train_casenet2d.py --config /mnt/lustre/shenrui/project/edgeDL/resources/train_config_case.yaml >./log/train-$now.log 2>&1 &
