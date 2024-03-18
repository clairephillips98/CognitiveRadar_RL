#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=4:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test.txt
#SBATCH --partition=compute_full_node
#SBATCH --ntasks=4
module load anaconda3
echo 1
source activate pytorch_env

echo next 4
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --use_double=0 --use_dueling=0 --use_noisy=0 --use_per=0 --use_n_steps=0 --cdl=0.9 --epsilon_init=1 &
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=2 --use_double=0 --use_dueling=0 --use_noisy=0 --use_per=0 --use_n_steps=0 --cdl=0.9 --epsilon_init=1 &
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --use_double=0 --use_dueling=0 --use_noisy=0 --use_per=0 --use_n_steps=1 --cdl=0.9 --epsilon_init=1 &
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --use_double=1 --use_dueling=0 --use_noisy=0 --use_per=0 --use_n_steps=0 --cdl=0.9 --epsilon_init=1
wait
echo next 4
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=2 --use_double=1 --use_dueling=0 --use_noisy=0 --use_per=0 --use_n_steps=0 --cdl=0.9 --epsilon_init=1 &
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=2 --use_double=0 --use_dueling=1 --use_noisy=0 --use_per=0 --use_n_steps=0 --cdl=0.9 --epsilon_init=1 &
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --use_double=0 --use_dueling=1 --use_noisy=0 --use_per=0 --use_n_steps=0 --cdl=0.9 --epsilon_init=1 &
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=2 --use_double=0 --use_dueling=0 --use_noisy=1 --use_per=0 --use_n_steps=0 --cdl=0.9 --epsilon_init=1
wait
echo next 4
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --use_double=0 --use_dueling=0 --use_noisy=1 --use_per=0 --use_n_steps=0 --cdl=0.9 --epsilon_init=1 &
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=2 --use_double=0 --use_dueling=0 --use_noisy=0 --use_per=0 --use_n_steps=1 --cdl=0.9 --epsilon_init=1
wait
