#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=8:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test.txt

module load anaconda3
echo 1
source activate pytorch_env
echo use_per 1
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --use_double=0 --use_dueling=0 --use_noisy=0 --use_per=1 --use_n_steps=0
