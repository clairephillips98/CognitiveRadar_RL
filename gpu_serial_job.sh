#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=8:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test.txt

module load anaconda3
echo 1
source activate pytorch_env
echo everything 1
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1
echo everything 0
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=0
everything 2
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=2
echo nothing 0
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=0 --use_double=0 --use_dueling=0 --use_noisy=0 --use_per=0 --use_n_steps=0
echo nothing 2
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=2 --use_double=0 --use_dueling=0 --use_noisy=0 --use_per=0 --use_n_steps=0
echo use_double 0
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=0 --use_double=1 --use_dueling=0 --use_noisy=0 --use_per=0 --use_n_steps=0
echo use_double 2
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=2 --use_double=1 --use_dueling=0 --use_noisy=0 --use_per=0 --use_n_steps=0
echo use_dueling 0
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=0 --use_double=0 --use_dueling=1 --use_noisy=0 --use_per=0 --use_n_steps=0
echo use_dueling 2
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=2 --use_double=0 --use_dueling=1 --use_noisy=0 --use_per=0 --use_n_steps=0
echo use_noisy 0
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=0 --use_double=0 --use_dueling=0 --use_noisy=1 --use_per=0 --use_n_steps=0
echo use_noisy 2
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=2 --use_double=0 --use_dueling=0 --use_noisy=1 --use_per=0 --use_n_steps=0
echo use_per 0
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=0 --use_double=0 --use_dueling=0 --use_noisy=0 --use_per=1 --use_n_steps=0
echo use_per 2
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=2 --use_double=0 --use_dueling=0 --use_noisy=0 --use_per=1 --use_n_steps=0
echo use_n_steps 0
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=0 --use_double=0 --use_dueling=0 --use_noisy=0 --use_per=0 --use_n_steps=1
echo use_n_steps 2
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=2 --use_double=0 --use_dueling=0 --use_noisy=0 --use_per=0 --use_n_steps=1