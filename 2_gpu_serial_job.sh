#!/bin/bash
# bests for longer experiment
full_name="track"
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
      --env_name="T5_A15" \
      --penalize_no_movement=1 \
      --radars=2 \
      --agents=1 \
      --baseline=$bl \
      --outside_radar_value=0.2 \
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
      --env_name="stochastic{hd}" \
      --penalize_no_movement=1 \
      --radars=1 \
      --agents=1 \
      --baseline=0 \
      --outside_radar_value=0.0 \
      --blur_sigma=0.5 \
      --relative_change=0 \
      --hidden_dim=$hd \
      --speed_scale=2 \
      --max_train_steps=10000000
EOT
done
for hd in 128 256; do
  for marl in "some_shared_info" "some_shared_info_shared_reward" "shared_targets_only" "single_agent"; do
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
      --env_name="stochastic${hd}" \
      --penalize_no_movement=1 \
      --radars=2 \
      --agents=2 \
      --baseline=0 \
      --outside_radar_value=1.0 \
      --blur_sigma=0.5 \
      --relative_change=0 \
      --hidden_dim=$hd \
      --type_of_MARL=$marl \
      --max_train_steps=10000000
EOT
  done
done
for hd in 128 156; do
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
      --env_name="stochastic{hd}" \
      --penalize_no_movement=1 \
      --radars=2 \
      --agents=1 \
      --baseline=0 \
      --outside_radar_value=1.0 \
      --blur_sigma=0.5 \
      --relative_change=0 \
      --hidden_dim=$hd \
      --type_of_MARL=$marl \
      --max_train_steps=10000000
EOT