for hl in 128 264 ; do #hidden layer
  for marl in "some_shared_info" "some_shared_info_shared_reward" "shared_targets_only" "single_agent"; do
    full_name="TRACK_REDOING_FINAL_hd${hl}"
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
      --outside_radar_value=1\
      --blur_sigma=0.3 \
      --relative_change=0 \
      --use_noisy=1 \
      --speed_scale=2 \
      --hidden_dim=$hl \
      --n_steps=1 \
      --max_train_steps=3000000 \
      --type_of_MARL=$marl \
      --evaluate_freq=500\
      --evaluate_times=2
EOT
  done
done