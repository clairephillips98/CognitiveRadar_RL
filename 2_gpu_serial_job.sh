#!/bin/bash
# bests for longer experiment
for bs in 0.5; do
  for ss in 2 3; do
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
      --agents=2 \
      --baseline=0 \
      --outside_radar_value=0.2 \
      --blur_sigma=0.5 \
      --relative_change=0 \
      --speed_scale=2 \
      --MARL="some_shared_info" \
      --max_train_steps=4000000
EOT
  done
done