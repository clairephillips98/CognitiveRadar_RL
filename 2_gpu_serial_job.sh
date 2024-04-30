#!/bin/bash

# different bs experiment
single_agent_Rainbow_
DQN_env_a26_reward_os_0.6_
nstep_1_hd_64_common_destination_
1.0_odds_n1_br1_se23_bs0.5_ss1_sl0
_a1_pnm1_rc1_r1_a1
for nstep in 0 1 3 5 7 ; do # blur sigma
full_name='nstep_experiments_a29'
  sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=9:00:0
#SBATCH --export=ALL
#SBATCH --output=experiment${bs}_2.txt
module load anaconda3
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main \
    --blur_radius=1 \
    --cdl=1 \
    --epsilon_init=0.5 \
    --load_model=0 \
    --env_name="${full_name}${nstep}" \
    --penalize_no_movement=1 \
    --radars=1 \
    --agents=1 \
    --baseline=0 \
    --outside_radar_value=0.5 \
    --blur_sigma=0.5 \
    --relative_change=0 \
    --speed_scale=1 \
    --hidden_dim=128 \
    --n_steps=$nstep \
    --max_train_steps=2000000
EOT
done