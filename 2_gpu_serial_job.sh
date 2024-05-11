#!/bin/bash
# bests for longer experiment
for bl in 1 2 3 4 5; do
  for r in 1 2; do
    full_name="V2_T5_a15"
    sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:0
#SBATCH --export=ALL
#SBATCH --output=run.txt
module load anaconda3
source activate pytorch_env

python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main \
      --blur_radius=1 \
      --cdl=0 \
      --epsilon_init=0.5 \
      --load_model=0 \
      --env_name="${full_name}" \
      --penalize_no_movement=1 \
      --radars=$r \
      --agents=1 \
      --baseline=$bl \
      --outside_radar_value=0.2 \
      --blur_sigma=0.3 \
      --relative_change=0 \
      --speed_scale=2 \
      --max_train_steps=1000000
EOT
    sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:0
#SBATCH --export=ALL
#SBATCH --output=run.txt
module load anaconda3
source activate pytorch_env

python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main \
      --blur_radius=1 \
      --cdl=0 \
      --epsilon_init=0.5 \
      --load_model=0 \
      --env_name="${full_name}" \
      --penalize_no_movement=1 \
      --radars=$r \
      --agents=1 \
      --baseline=$bl \
      --outside_radar_value=0.2 \
      --blur_sigma=0.5 \
      --relative_change=0 \
      --speed_scale=2 \
      --max_train_steps=1000000
EOT
  done
done