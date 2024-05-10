#!/bin/bash
# bests for longer experiment

for bs in 0.3 0.5; do
  for ss in 2; do
      full_name="tester"
      sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=0:30:0
#SBATCH --export=ALL
#SBATCH --output=run.txt
module load anaconda3
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main \
      --blur_radius=1 \
      --cdl=0 \
      --epsilon_init=0.5 \
      --load_model=0 \
      --env_name="${full_name}128" \
      --penalize_no_movement=1 \
      --radars=2 \
      --agents=1 \
      --baseline=0 \
      --outside_radar_value=0.2 \
      --blur_sigma=$bs \
      --relative_change=0 \
      --use_noisy=1 \
      --speed_scale=$ss \
      --hidden_dim=128 \
      --n_steps=1 \
      --type_of_MARL="some_shared_info_shared_reward"
      --max_train_steps=4000000
  done
done