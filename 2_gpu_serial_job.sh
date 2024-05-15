#!/bin/bash
# bests for longer experiment
full_name="V2_T5_a15_ns1_hd"
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
      --env_name="V2_T5_a15_ns1_hd256" \
      --penalize_no_movement=1 \
      --radars=2 \
      --agents=1 \
      --baseline=0 \
      --hidden_dim=256 \
      --outside_radar_value=0.0 \
      --blur_sigma=0.5 \
      --relative_change=0 \
      --speed_scale=2 \
      --type_of_MARL="single_agent" \
      --max_train_steps=4000000
EOT
for marl in "some_shared_info" "some_shared_info_shared_reward" "shared_targets_only"; do
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
      --env_name="V2_T5_a15_ns1_hd256" \
      --penalize_no_movement=1 \
      --radars=2 \
      --agents=2 \
      --baseline=0 \
      --outside_radar_value=0.0 \
      --blur_sigma=0.5 \
      --relative_change=0 \
      --hidden_dim=256 \
      --speed_scale=2 \
      --type_of_MARL=$marl \
      --max_train_steps=4000000
EOT
done
for bl in 1 2 3 5; do
    sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:0
#SBATCH --export=ALL
#SBATCH --output=run.txt
module load anaconda3
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main \
      --blur_radius=1 \
      --cdl=0 \
      --load_model=1 \
      --env_name="V2_T5_a15_ns1_hd256" \
      --penalize_no_movement=1 \
      --radars=2 \
      --agents=1 \
      --baseline=$bl \
      --outside_radar_value=0.0 \
      --blur_sigma=0.5 \
      --relative_change=0 \
      --speed_scale=2 \
      --max_train_steps=500000
EOT
done
for bl in 1 2 3 5; do
    sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:0
#SBATCH --export=ALL
#SBATCH --output=run.txt
module load anaconda3
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main \
      --blur_radius=1 \
      --cdl=0 \
      --load_model=1 \
      --env_name="V2_T5_a15_ns1_hd256" \
      --penalize_no_movement=1 \
      --radars=1 \
      --agents=1 \
      --baseline=$bl \
      --outside_radar_value=0.0 \
      --blur_sigma=0.5 \
      --relative_change=0 \
      --speed_scale=2 \
      --max_train_steps=500000
EOT
done
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:0
#SBATCH --export=ALL
#SBATCH --output=run.txt
module load anaconda3
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main \
      --blur_radius=1 \
      --cdl=0 \
      --load_model=1 \
      --env_name="V2_T5_a15_ns1_hd256" \
      --penalize_no_movement=1 \
      --radars=1 \
      --agents=1 \
      --baseline=0 \
      --outside_radar_value=0.0 \
      --blur_sigma=0.5 \
      --relative_change=0 \
      --speed_scale=2 \
      --hidden_dim=256 \
      --max_train_steps=4000000
EOT
