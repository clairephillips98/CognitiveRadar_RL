#!/bin/bash


# Iterate over the sequence of floating-point numbers


for os in 0.8 0.7; do
  for bl in 1 2 3 4 5; do #bl
        full_name="a25_summed_reward_os_${os}"
        echo "$full_name"
        sbatch <<EOT
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr16_${i}.txt

module load anaconda3
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main \
    --blur_radius=1 \
    --cdl=1 \
    --epsilon_init=0.5 \
    --load_model=0 \
    --env_name="$full_name" \
    --penalize_no_movement=1 \
    --radars=1 \
    --agents=1 \
    --baseline=$bl \
    --outside_radar_value=$os \
    --blur_sigma=0.5 \
    --relative_change=0 \
    --speed_scale=1
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main \
    --blur_radius=1 \
    --cdl=1 \
    --epsilon_init=0.5 \
    --load_model=0 \
    --env_name="$full_name" \
    --penalize_no_movement=1 \
    --radars=2 \
    --agents=1 \
    --baseline=$bl \
    --outside_radar_value=$os \
    --blur_sigma=0.5 \
    --relative_change=0 \
    --speed_scale=1
EOT
  done
  for m in 32 64 128; do
    for n in 1 5 7; do
      for rc in 0 1; do
        name='a25_summed_reward'
        full_name="${name}_os${os}_nstep_${n}_hd_${m}"
        echo "$full_name"
        sbatch <<EOT
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=8:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr16_${i}.txt

module load anaconda3
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main \
    --blur_radius=1 \
    --cdl=1 \
    --epsilon_init=0.5 \
    --load_model=0 \
    --env_name="$full_name" \
    --penalize_no_movement=1 \
    --radars=1 \
    --agents=1 \
    --baseline=0 \
    --outside_radar_value=$os \
    --blur_sigma=0.5 \
    --relative_change=$rc \
    --speed_scale=1 \
    --hidden_dim=$m \
    --n_steps=$n
EOT
      done
    done
  done
done

for os in 0.8 0.7; do
  for m in 64 128; do
    for rc in 0 1; do
      name='a25_summed_reward'
      full_name="${name}_os${os}_nstep_${n}_hd_${m}"
      sbatch <<EOT
      #!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=8:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr16_${i}.txt

module load anaconda3
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main \
    --blur_radius=1 \
    --cdl=1 \
    --epsilon_init=0.5 \
    --load_model=0 \
    --env_name="$full_name" \
    --penalize_no_movement=1 \
    --radars=2 \
    --agents=1 \
    --baseline=0 \
    --outside_radar_value=$os \
    --blur_sigma=0.5 \
    --relative_change=$rc \
    --speed_scale=1 \
    --hidden_dim=$m \
    --n_steps=1
EOT
      for t in 'some_shared_info' 'some_shared_info_shared_reward' 'shared_targets_only' 'single_agent'; do
        echo "$full_name"
        sbatch <<EOT
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=8:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr16_${i}.txt

module load anaconda3
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main \
    --blur_radius=1 \
    --cdl=1 \
    --epsilon_init=0.5 \
    --load_model=0 \
    --env_name="$full_name" \
    --penalize_no_movement=1 \
    --radars=2 \
    --agents=2 \
    --baseline=0 \
    --outside_radar_value=$os \
    --blur_sigma=0.5 \
    --relative_change=$rc \
    --speed_scale=1 \
    --hidden_dim=$m \
    --n_steps=1 \
    --type_of_MARL=$t
EOT
      done
    done
  done
done