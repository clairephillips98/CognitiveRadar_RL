#!/bin/bash

# Iterate over the sequence of floating-point numbers

for bl in 1 2 3 4 5; do #bl
    for i in 1 3 5; do #ss
      for r in 1; do #radars
            name='a19_penalty_airport_cond_a8_t30_unmask_0.1_'
            full_name="${name}r${r}_bl${bl}_"
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
        for m in 64 128 256; do
          for n in 1 3 5; do
            name='a19_penalty_airport_cond_a8_t30_unmask_0.1_'
            full_name="${name}r${r}_bl${bl}_nstep_${n}_hd_${m}"
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
    --env_name="$full_name" \
    --penalize_no_movement=1 \
    --radars=1 \
    --agents=1 \
    --baseline=$bl \
    --outside_radar_value=0.9 \
    --blur_sigma=0.5 \
    --relative_change=0 \
    --speed_scale=$i \
    --hidden_dim=$m \
    --n_steps=$n
EOT
            done
        done
    done
done