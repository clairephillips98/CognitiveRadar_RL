#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=8:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_mar26.txt
module load anaconda3
echo 1
source activate pytorch_env

python -m rl_agents.baseline_model --cdl=1 --speed_layer=1 --env_name='_airport_chance_of_no_detect_8_actions_speed_layer_speed_reward' --speed_scale=0 --speed_layer=1
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --cdl=1 --epsilon_init=1 --epsilon_init=0.5 --load_model=0 --env_name='_airport_chance_of_no_detect_8_actions_speed_layer_speed_scale_100' --speed_scale=0 --speed_layer=1 --speed_scale=100
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --use_double=0 --use_dueling=0 --use_noisy=1 --use_per=0 --use_n_steps=0 --cdl=1 --epsilon_init=0.5 --load_model=0 --env_name='_airport_chance_of_no_detect_8_actions_speed_layer_speed_scale_100' --speed_scale=0 --speed_layer=1 --speed_scale=100
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --use_double=0 --use_dueling=0 --use_noisy=0 --use_per=1 --use_n_steps=0 --cdl=1 --epsilon_init=0.5 --load_model=0 --env_name='_airport_chance_of_no_detect_8_actions_speed_layer_speed_scale_100' --speed_scale=0 --speed_layer=1 --speed_scale=100
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main -use_double=0 --use_dueling=0 --use_noisy=0 --use_per=0 --use_n_steps=1 --cdl=1  --epsilon_init=0.5 --load_model=0 --env_name='_airport_chance_of_no_detect_8_actions_speed_layer_speed_scale_100' --speed_scale=0 --speed_layer=1 --speed_scale=100
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --use_double=1 --use_dueling=0 --use_noisy=0 --use_per=0 --use_n_steps=0 --cdl=1  --epsilon_init=0.5 --load_model=0 --env_name='_airport_chance_of_no_detect_8_actions_speed_layer_speed_scale_100' --speed_scale=0 --speed_layer=1 --speed_scale=100
