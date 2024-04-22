#!/bin/bash
# Iterate over the sequence of floating-point numbers
for b in 0.3 0.5; do # blur sigma
  for l in 0.95; do # area outside of mask
    name='a19_penalty_airport_cond_a8_t30_unmask_'$l''
    echo $name
    for i in {1..2}; do # radars
      for j in {1..2}; do
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr16_${i}.txt
module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --cdl=1 --epsilon_init=0.5 --load_model=0 --speed_scale=1 --env_name="$name" --penalize_no_movement=1 --radars=$i --agents=1 --baseline=$j --outside_radar_value=$l --blur_sigma=$b
EOT
      done
    done
  done
done
