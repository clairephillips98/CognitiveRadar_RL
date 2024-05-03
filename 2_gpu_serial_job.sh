#!/bin/bash

# different bs experiment
full_name="faster_quarter_view_a15"
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:0
#SBATCH --export=ALL
#SBATCH --output=bl_2.txt
module load anaconda3
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main \
    --blur_radius=1 \
    --epsilon_init=0.5 \
    --load_model=0 \
    --env_name="${full_name}" \
    --penalize_no_movement=1 \
    --radars=1 \
    --agents=1 \
    --baseline=1 \
    --outside_radar_value=0.2 \
    --blur_sigma=0.5 \
    --relative_change=0\
    --speed_scale=1 \
    --max_train_steps=200000
EOT