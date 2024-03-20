#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:0
#SBATCH --export=ALL
#SBATCH --output=testing.txt
#SBATCH --ntasks=4
module load anaconda3
echo 1
source activate pytorch_env

echo models
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --use_double=0 --use_dueling=1 --use_noisy=0 --use_per=1 --use_n_steps=0 --cdl=1 --epsilon_init=0.5
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --use_double=0 --use_dueling=1 --use_noisy=1 --use_per=0 --use_n_steps=0 --cdl=1 --epsilon_init=0.5
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --use_double=0 --use_dueling=0 --use_noisy=0 --use_per=1 --use_n_steps=1 --cdl=1 --epsilon_init=0.5
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --use_double=0 --use_dueling=1 --use_noisy=0 --use_per=0 --use_n_steps=1 --cdl=1  --epsilon_init=0.5
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --use_double=1 --use_dueling=0 --use_noisy=0 --use_per=0 --use_n_steps=1 --cdl=1  --epsilon_init=0.5
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --cdl=1 --epsilon_init=0.5
echo baselines
python -m rl_agents.baseline_model --baseline_model_type='min_variance' --cdl=1 --gpu_number=0
python -m rl_agents.baseline_model --baseline_model_type='max_variance' --cdl=1 --gpu_number=1
python -m rl_agents.baseline_model --baseline_model_type='simple' --cdl=1 --gpu_number=2