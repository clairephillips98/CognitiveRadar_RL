#!/bin/bash
name='apr_14_penalty_airport_cond_a8_t30_r2_e0.5'
radars=2

sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=9:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr12_0.txt
module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.baseline_model --env_name="$name" --speed_scale=1 --radars=2 --baseline_model_type="simple" --penalize_no_movement=1
EOT
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=9:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr12_1.txt
module load anaconda3
echo 1
source activate pytorch_env
# 2 radars, nm, pnm
python -m rl_agents.baseline_model --env_name="$name" --speed_scale=1 --radars=2 --baseline_model_type="no_movement" --penalize_no_movement=1
EOT

sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=9:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr12_4.txt
module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.baseline_model --env_name="$name" --speed_scale=1 --radars=1 --baseline_model_type="no_movement" --penalize_no_movement=1
EOT
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=9:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr12_5.txt
module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.baseline_model --env_name="$name" --speed_scale=1 --radars=1 --baseline_model_type="simple" --penalize_no_movement=1
EOT