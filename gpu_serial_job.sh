#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --gpus-per-node=1
#SBATCH --mem=4000M               # memory per node
#SBATCH --time=0-03:00
chmod -x create_env.sh
sh create_env.sh                         # you can use 'nvidia-smi' for a test