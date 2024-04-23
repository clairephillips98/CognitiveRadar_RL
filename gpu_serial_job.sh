#!/bin/bash

# Iterate over the sequence of floating-point numbers
for r in 1 2; do #radars
    for bl in 1 2 3 4 5; do #bl
        for i in 1 3 5; do #ss
            name='a19_penalty_airport_cond_a8_t30_unmask_0.1_'
            full_name="${name}r${r}_bl${bl}_ss${i}"
            echo "$full_name"

            sbatch <<EOT
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:0
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
    --env_name="$full_name" \
    --penalize_no_movement=1 \
    --radars=$r \
    --agents=1 \
    --baseline=$bl \
    --outside_radar_value=0.9 \
    --blur_sigma=0.5 \
    --relative_change=0 \
    --speed_scale=$i
EOT
        done
    done
done