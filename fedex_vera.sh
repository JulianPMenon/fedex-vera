#!/bin/bash
#SBATCH --job-name=fedex_vera
#SBATCH --output=logs/fedex_vera_client_%A_%a.out
#SBATCH --error=logs/fedex_vera_client_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL 
#SBATCH --gres=gpu:4

# Increase file descriptor limit for PyTorch multiprocessing
ulimit -n 65536

module load conda
module load cuda

conda activate fed_vera  

echo "===== SLURM DIAGNOSTICS ====="
hostname
nvidia-smi
which python
python --version
python - <<EOF
import numpy as np
import torch

print("numpy version:", np.__version__)
print("ndarray exists:", hasattr(np, "ndarray"))

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
EOF

echo "============================"

# Run your FedVera script for this client
# FedEX VeRA - Parallel client training on separate GPUs
# Server on GPU 0, Clients on GPUs 1,2,3 (no contention)
python fed_train_glue.py --model=models/models--FacebookAI--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b --task=rte --agg_type=ours_vera --vera --num_clients=3 --r=128 --rounds=51 --lr=1e-3 --local_epochs=5 --parallel_clients --server_gpu 0 --client_gpus 1,2,3
