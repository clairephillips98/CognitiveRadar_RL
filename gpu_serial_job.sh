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
echo 4
python3 -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main
echo everything
python3 -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main
echo nothing
python3 -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --use_double=0 --use_dueling=0 --use_noisy=0 --use_per=0 --use_n_steps=0
echo use_double
python3 -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --use_double=1 --use_dueling=0 --use_noisy=0 --use_per=0 --use_n_steps=0
echo use_dueling
python3 -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --use_double=0 --use_dueling=1 --use_noisy=0 --use_per=0 --use_n_steps=0
echo use_noisy
python3 -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --use_double=0 --use_dueling=0 --use_noisy=1 --use_per=0 --use_n_steps=0
echo use_per
python3 -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --use_double=0 --use_dueling=0 --use_noisy=0 --use_per=1 --use_n_steps=0
echo use_n_steps
python3 -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --use_double=0 --use_dueling=0 --use_noisy=0 --use_per=0 --use_n_steps=1
