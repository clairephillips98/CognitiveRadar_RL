#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test.txt

module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main
echo everything
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main
echo nothing
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --use_double=0 --use_dueling=0 --use_noisy=0 --use_per=0 --use_n_steps=0
echo use_double
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --use_double=1 --use_dueling=0 --use_noisy=0 --use_per=0 --use_n_steps=0
echo use_dueling
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --use_double=0 --use_dueling=1 --use_noisy=0 --use_per=0 --use_n_steps=0
echo use_noisy
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --use_double=0 --use_dueling=0 --use_noisy=1 --use_per=0 --use_n_steps=0
echo use_per
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --use_double=0 --use_dueling=0 --use_noisy=0 --use_per=1 --use_n_steps=0
echo use_n_steps
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --use_double=0 --use_dueling=0 --use_noisy=0 --use_per=0 --use_n_steps=1
echo baseline
python -m rl_agents.baseline_model --baseline_model_type 'min_variance'
python -m rl_agents.baseline_model --baseline_model_type 'max_variance'
python -m rl_agents.baseline_model --baseline_model_type 'simple'