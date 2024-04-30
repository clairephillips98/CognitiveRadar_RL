#!/bin/bash

# rerun best for longer?

full_name='a26_v2s'
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:0
#SBATCH --export=ALL
#SBATCH --output=experiment${bs}_2.txt
module load anaconda3
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main \
    --blur_radius=1 \
    --cdl=1 \
    --epsilon_init=0.5 \
    --load_model=0 \
    --env_name="${full_name}" \
    --penalize_no_movement=1 \
    --radars=1 \
    --agents=1 \
    --baseline=0 \
    --outside_radar_value=0.6 \
    --blur_sigma=0.5 \
    --relative_change=1 \
    --speed_scale=1 \
    --hidden_dim=64 \
    --n_steps=$nstep \
    --max_train_steps=2000000
EOT

sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:0
#SBATCH --export=ALL
#SBATCH --output=experiment${bs}_2.txt
module load anaconda3
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main \
    --blur_radius=1 \
    --cdl=1 \
    --epsilon_init=0.5 \
    --load_model=0 \
    --env_name="${full_name}" \
    --penalize_no_movement=1 \
    --radars=2 \
    --agents=1 \
    --baseline=0 \
    --outside_radar_value=0.6 \
    --blur_sigma=0.5 \
    --relative_change=0 \
    --speed_scale=1 \
    --hidden_dim=128 \
    --n_steps=$nstep \
    --max_train_steps=2000000
EOT


sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:0
#SBATCH --export=ALL
#SBATCH --output=experiment${bs}_2.txt
module load anaconda3
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main \
    --blur_radius=1 \
    --cdl=1 \
    --epsilon_init=0.5 \
    --load_model=0 \
    --env_name="${full_name}" \
    --penalize_no_movement=1 \
    --radars=2 \
    --agents=1 \
    --baseline=0 \
    --outside_radar_value=0.6 \
    --blur_sigma=0.5 \
    --relative_change=0 \
    --speed_scale=1 \
    --hidden_dim=64 \
    --n_steps=$nstep \
    --max_train_steps=2500000
EOT

sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:0
#SBATCH --export=ALL
#SBATCH --output=experiment${bs}_2.txt
module load anaconda3
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main \
    --blur_radius=1 \
    --cdl=1 \
    --epsilon_init=0.5 \
    --load_model=0 \
    --env_name="${full_name}" \
    --penalize_no_movement=1 \
    --radars=2 \
    --agents=2 \
    --baseline=0 \
    --outside_radar_value=0.6 \
    --blur_sigma=0.5 \
    --relative_change=0 \
    --speed_scale=1 \
    --hidden_dim=64 \
    --n_steps=$nstep \
    --max_train_steps=2500000
EOT