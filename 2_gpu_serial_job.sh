#!/bin/bash
# bests for longer experiment
for bl in 1 2 3 4 5; do
  for r in 1 2; do
    full_name="V2_T5_a15"
    sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=8:00:0
#SBATCH --export=ALL
#SBATCH --output=run.txt
module load anaconda3
source activate pytorch_env

mkdir -p $SCRATCH/nvidia-mps $SCRATCH/nvidia-log
export CUDA_MPS_PIPE_DIRECTORY=$SCRATCH/nvidia-mps CUDA_MPS_LOG_DIRECTORY=$SCRATCH/nvidia-log
nvidia-cuda-mps-control -d

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
      --max_train_steps=200000 & python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main \
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
      --max_train_steps=200000
EOT
  done
done

for bs in 0.3 0.5; do
  for ss in 2; do
      full_name="V2_T5_a15_ns1_hd128"
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
EOT
  done
done