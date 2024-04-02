#!/bin/bash
name='exp3_airport_cond_a8_t30_r2_e1'
radars=2
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=9:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr2_5.txt
module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --cdl=1 --epsilon_init=1 --load_model=0 --env_name="$name" --speed_layer=1 --speed_scale=1 --agents=1 --radars=2
EOT
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=9:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr2_6.txt
module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --cdl=1 --epsilon_init=1 --load_model=0 --env_name="$name" --speed_layer=1 --speed_scale=1 --agents=2 --radars=2
EOT
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=9:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr2_7.txt
module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.baseline_model --cdl=1 --env_name="$name" --speed_scale=1 --radars=2
EOT
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=9:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr2_8.txt
module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.baseline_model --cdl=1 --env_name="$name" --speed_scale=1 --radars=2 --baseline_model_type="no_movement"
EOT
