#!/bin/bash

# Iterate over the sequence of floating-point numbers
for r in 0 1; do #relative
    for i in 0 1 5; do #ss
        for m in 64 128 256; do #hidden dim
          for n in 1 3 5; do #nstep
            name='a19_penalty_airport_cond_a8_t30_unmask_0.1_'
            echo "name"

            sbatch <<EOT
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=6:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr16_${i}.txt

module load anaconda3
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main \
    --blur_radius=1 \
    --cdl=1 \
    --epsilon_init=0.5 \
    --load_model=0 \
    --speed_scale=1 \
    --env_name="name" \
    --penalize_no_movement=1 \
    --radars=1 \
    --agents=1 \
    --baseline=0 \
    --outside_radar_value=0.9 \
    --blur_sigma=0.5 \
    --relative_change=$r \
    --speed_scale=$i
    --n_steps=$n \
    --hidden_dim=$m \
EOT
        done
    done
done