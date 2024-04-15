#!/bin/bash
name='a15_penalty_airport_cond_a8_t30_r2_e0.5'
radars=2

sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=9:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr15_0.txt
module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.baseline_model --env_name="$name" --speed_scale=1 --radars=2 --baseline_model_type="simple" --penalize_no_movement=1
EOT