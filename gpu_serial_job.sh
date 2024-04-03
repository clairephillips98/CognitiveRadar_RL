#!/bin/bash
name='improve_movement_trials_airport_cond_a8_t30_r2_e0.5'
radars=2

sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=9:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr3_5.txt
module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.baseline_model --env_name="$name" --speed_scale=1 --radars=2 --baseline_model_type="simple"
EOT
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=9:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr3_6.txt
module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.baseline_model --env_name="$name" --speed_scale=1 --radars=2 --baseline_model_type="no_movement"
EOT
