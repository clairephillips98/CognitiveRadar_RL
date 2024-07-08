for os in 0; do
  for hl in 528 1048; do #hidden layer
    for marl in "shared_targets_only"; do
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
      --load_model=1 \
      --env_name="${full_name}" \
      --penalize_no_movement=1 \
      --radars=2 \
      --agents=2 \
      --baseline=0 \
      --outside_radar_value=$os \
      --blur_sigma=0.2 \
      --relative_change=0 \
      --use_noisy=1 \
      --speed_scale=1000 \
      --hidden_dim=$hl \
      --n_steps=1 \
      --max_train_steps=3000000 \
      --type_of_MARL=$marl
EOT
    done
  done
done
