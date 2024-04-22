#!/bin/bash
# Iterate over the sequence of floating-point numbers
for m in 0 1 5; do #speed scale
  for l in 64 128 256; do #hidden dim size
    name='a19_penalty_airport_cond_a8_t30_unmask_0.1'
    echo $name
    for k in 1 3 5; do # n steps
          for j in 0 1; do #step types
    name='a19_penalty_airport_cond_a8_t30_unmask_0.1_nstep${k}_hd${l}_'
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=6:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr16_${i}.txt
module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --cdl=1 --epsilon_init=0.5 --load_model=0 --speed_scale=1 --env_name="$name" --penalize_no_movement=1 --radars=2 --agents=2 --baseline=0 --outside_radar_value=0.9 --blur_sigma=0.5 --relative_change=$j --n_steps=$k --hidden_dim=$l --speed_scale=$m --type_of_MARL='some_shared_info_shared_reward'
EOT
      done
    done
  done
done
