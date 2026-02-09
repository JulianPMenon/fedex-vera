#!/bin/bash
#SBATCH --job-name=fedvera_parallel
#SBATCH --output=logs/fedvera_client_%A_%a.out
#SBATCH --error=logs/fedvera_client_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --time=08:00:00
#SBATCH --mail-type=ALL 
#SBATCH --gpus=1

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
python fed_train_glue.py --model=models/models--FacebookAI--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b --task=cola --agg_type=ours_vera --vera --num_clients=3 --r=120 --rounds=3 --lr=1e-3 --local_epochs=80 
# Run your FedEXLora script for this client
# python fed_train_glue.py --model=models/models--FacebookAI--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b --task=cola --agg_type=ours --num_clients=3 --r=6 --rounds=3 --lr=1e-3 --local_epochs=80  
# Or, if you use a different entry point:
# python fed_agg.py --client_id $CLIENT_ID