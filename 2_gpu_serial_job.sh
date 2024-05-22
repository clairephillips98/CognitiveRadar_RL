#!/bin/bash
# bests for longer experiment
full_name="search"
for bl in 1 2 3 4 5; do
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
      --load_model=1 \
      --env_name="20pure_search_a8256" \
      --penalize_no_movement=1 \
      --radars=1 \
      --agents=1 \
      --baseline=$bl \
      --outside_radar_value=1.0 \
      --blur_sigma=0.5 \
      --relative_change=0 \
      --speed_scale=2 \
      --max_train_steps=400000
EOT
done
for hd in 128 256; do
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
      --load_model=1 \
      --env_name="search${hd}" \
      --penalize_no_movement=1 \
      --radars=1 \
      --agents=1 \
      --baseline=0 \
      --outside_radar_value=0.0 \
      --blur_sigma=0.5 \
      --relative_change=0 \
      --hidden_dim=$hd \
      --speed_scale=0 \
      --max_train_steps=10000000
EOT
done
