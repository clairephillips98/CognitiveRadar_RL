#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:0
#SBATCH --export=ALL
#SBATCH --output=$SCRATCH/cphil_test.txt
module load anaconda3
echo 1
source activate pytorch_env
echo 2
echo $PYTHONPATH
echo 2
pip list
echo 3
conda list
python -m check_packages.py 
python3 -m rl_agents.baseline_model
echo 11
python3 -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main
 

