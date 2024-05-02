#!/bin/bash

# different bs experiment
full_name="quarter_view_a15"
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=3:00:0
#SBATCH --export=ALL
#SBATCH --output=bl_2.txt
module load anaconda3
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main \
    --blur_radius=1 \
    --epsilon_init=0.5 \
    --load_model=0 \
    --env_name="${full_name}" \
    --penalize_no_movement=1 \
    --radars=1 \
    --agents=1 \
    --baseline=1 \
    --outside_radar_value=0.2 \
    --blur_sigma=0.3 \
    --relative_change=0\
    --speed_scale=1 \
    --max_train_steps=200000
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main \
    --blur_radius=1 \
    --cdl=0 \
    --epsilon_init=0.5 \
    --load_model=0 \
    --env_name="${full_name}" \
    --penalize_no_movement=1 \
    --radars=1 \
    --agents=1 \
    --baseline=3 \
    --outside_radar_value=0.2 \
    --blur_sigma=0.3 \
    --relative_change=0\
    --speed_scale=1 \
    --max_train_steps=200000
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main \
    --blur_radius=1 \
    --cdl=0 \
    --epsilon_init=0.5 \
    --load_model=0 \
    --env_name="${full_name}" \
    --penalize_no_movement=1 \
    --radars=1 \
    --agents=1 \
    --baseline=5 \
    --outside_radar_value=0.2 \
    --blur_sigma=0.3 \
    --relative_change=0\
    --speed_scale=1 \
    --max_train_steps=200000

python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main \
    --blur_radius=1 \
    --cdl=0 \
    --epsilon_init=0.5 \
    --load_model=0 \
    --env_name="${full_name}" \
    --penalize_no_movement=1 \
    --radars=2 \
    --agents=1 \
    --baseline=1 \
    --outside_radar_value=0.2 \
    --blur_sigma=0.3 \
    --relative_change=0\
    --speed_scale=1 \
    --max_train_steps=200000
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main \
    --blur_radius=1 \
    --cdl=0 \
    --epsilon_init=0.5 \
    --load_model=0 \
    --env_name="${full_name}" \
    --penalize_no_movement=1 \
    --radars=2 \
    --agents=1 \
    --baseline=3 \
    --outside_radar_value=0.2 \
    --blur_sigma=0.3 \
    --relative_change=0\
    --speed_scale=1 \
    --max_train_steps=200000
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main \
    --blur_radius=1 \
    --cdl=0 \
    --epsilon_init=0.5 \
    --load_model=0 \
    --env_name="${full_name}" \
    --penalize_no_movement=1 \
    --radars=2 \
    --agents=1 \
    --baseline=5 \
    --outside_radar_value=0.2 \
    --blur_sigma=0.3 \
    --relative_change=0\
    --speed_scale=1 \
    --max_train_steps=200000
EOT

sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=3:00:0
#SBATCH --export=ALL
#SBATCH --output=bl_2.txt
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
    --baseline=1 \
    --outside_radar_value=0.2 \
    --blur_sigma=0.5 \
    --relative_change=0\
    --speed_scale=1 \
    --max_train_steps=200000
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main \
    --blur_radius=1 \
    --cdl=0 \
    --epsilon_init=0.5 \
    --load_model=0 \
    --env_name="${full_name}" \
    --penalize_no_movement=1 \
    --radars=2 \
    --agents=1 \
    --baseline=3 \
    --outside_radar_value=0.2 \
    --blur_sigma=0.5 \
    --relative_change=0\
    --speed_scale=1 \
    --max_train_steps=200000
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main \
    --blur_radius=1 \
    --cdl=0 \
    --epsilon_init=0.5 \
    --load_model=0 \
    --env_name="${full_name}" \
    --penalize_no_movement=1 \
    --radars=2 \
    --agents=1 \
    --baseline=5 \
    --outside_radar_value=0.2 \
    --blur_sigma=0.5 \
    --relative_change=0\
    --speed_scale=1 \
    --max_train_steps=200000
EOT


for bs in 0.3 0.5 ; do #bs
  for hl in 64 128 ; do #hidden layer
    for nstep in 1 3 5 7 ; do # nstep
      for st in 0 1 ; do # step type
        full_name="quarter_view_a15_ns${nstep}_hd${hl}"
        sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=6:00:0
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
    --radars=1 \
    --agents=1 \
    --baseline=0 \
    --outside_radar_value=0.2 \
    --blur_sigma=$bs \
    --relative_change=$st \
    --speed_scale=1 \
    --hidden_dim=$hl \
    --n_steps=$nstep \
    --max_train_steps=3000000
EOT
      done
    done
    full_name="quarter_view_a15_ns1_hd${hl}"
    sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=8:00:0
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
    --outside_radar_value=0.2 \
    --blur_sigma=$bs \
    --relative_change=0 \
    --speed_scale=1 \
    --hidden_dim=$hl \
    --n_steps=1 \
    --max_train_steps=3000000
EOT
    for marl in "some_shared_info" "some_shared_info_shared_reward" "shared_targets_only" "single_agent"; do
      sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=8:00:0
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
    --outside_radar_value=0.2 \
    --blur_sigma=$bs \
    --relative_change=0 \
    --speed_scale=1 \
    --hidden_dim=$hl \
    --n_steps=1 \
    --type_of_MARL=$marl\
    --max_train_steps=3000000
EOT
    done
  done
done