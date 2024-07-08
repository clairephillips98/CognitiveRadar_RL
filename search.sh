for bl in 1 2 3 4 5 ; do #hidden layer
  for rad in 2; do
    full_name="NEW_SEARCH_REDOING_FINAL_hd${hl}"
    sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:0
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
      --outside_radar_value=0.2 \
      --blur_sigma=1 \
      --relative_change=0 \
      --use_noisy=1 \
      --speed_scale=0 \
      --n_steps=1 \
      --search_outer_circle=1 \
      --max_train_steps=3000000
EOT
  done
done