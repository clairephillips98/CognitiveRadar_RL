for os in 1.0; do
  for bl in 1 2 3 4 5 ; do #hidden layer
    for rad in 2; do
      full_name="NEW_TRACK_REDOING_FINAL_hd${hl}"
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
      --radars=$rad \
      --agents=1 \
      --baseline=$bl \
      --outside_radar_value=$os \
      --blur_sigma=0.2 \
      --relative_change=0 \
      --use_noisy=1 \
      --speed_scale=1000 \
      --n_steps=1 \
      --max_train_steps=3000000
EOT
    done
  done
  for hl in 128 264 528; do #hidden layer
      full_name="NEW_TRACK_REDOING_FINAL_hd${hl}"
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
      --radars=2 \
      --agents=1 \
      --baseline=0 \
      --outside_radar_value=$os \
      --blur_sigma=0.2 \
      --relative_change=0 \
      --use_noisy=1 \
      --speed_scale=10 \
      --hidden_dim=$hl \
      --n_steps=1 \
      --max_train_steps=3000000\
      --tracking_mode=1
EOT
  done
  for hl in 128 264 528; do #hidden layer
    for marl in "some_shared_info" "some_shared_info_shared_reward" "shared_targets_only" "single_agent"; do
      full_name="NEW_TRACK_REDOING_FINAL_hd${hl}"
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
      --radars=2 \
      --agents=2 \
      --baseline=0 \
      --outside_radar_value=$os \
      --blur_sigma=0.2 \
      --relative_change=0 \
      --use_noisy=1 \
      --speed_scale=10 \
      --hidden_dim=$hl \
      --n_steps=1 \
      --max_train_steps=3000000 \
      --type_of_MARL=$marl\
      --tracking_mode=1
EOT
    done
    full_name="NEW_TRACK_REDOING_FINAL_hd${hl}"
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
      --radars=2 \
      --agents=1 \
      --baseline=0 \
      --outside_radar_value=$os \
      --blur_sigma=0.2 \
      --relative_change=0 \
      --use_noisy=1 \
      --speed_scale=10 \
      --hidden_dim=$hl \
      --n_steps=1 \
      --max_train_steps=3000000\
      --tracking_mode=1
EOT
  done
done
