#!/bin/bash
#SBATCH --job-name=fedvera_parallel
#SBATCH --output=logs/fedvera_client_%A_%a.out
#SBATCH --error=logs/fedvera_client_%A_%a.err
#SBATCH --array=0-2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END 

# Activate conda environment
module load python/3.10
module load conda

conda activate fed_vera  



# Set the client id based on SLURM_ARRAY_TASK_ID
CLIENT_ID=$SLURM_ARRAY_TASK_ID

# Run your FedVera script for this client
python fed_train_glue.py --model=roberta-base --task=cola --agg_type=ours_vera --vera --num_clients=3 --r=4 --rounds=2 --lr=1e-3 --local_epochs=1 --client_id $CLIENT_ID
# Or, if you use a different entry point:
# python fed_agg.py --client_id $CLIENT_ID