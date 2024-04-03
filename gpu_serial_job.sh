#!/bin/bash
name='improve_movement_trials_airport_cond_a8_t30_r2_e0.5'
radars=2
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=9:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr3_1.txt
module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --cdl=1 --epsilon_init=0.5 --load_model=0 --env_name="$name" --speed_layer=0 --speed_scale=1 --agents=2 --radars=2
EOT
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=9:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr3_2.txt
module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --cdl=1 --epsilon_init=0.5 --load_model=0 --env_name="$name" --speed_layer=0 --speed_scale=1 --agents=1 --radars=2
EOT
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=9:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr3_3.txt
module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --cdl=1 --epsilon_init=0.5 --load_model=0 --env_name="$name" --speed_layer=0 --speed_scale=1 --agents=2 --radars=2 --relative_change=1
EOT
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=9:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr3_4.txt
module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --cdl=1 --epsilon_init=0.5 --load_model=0 --env_name="$name" --speed_layer=0 --speed_scale=1 --agents=1 --radars=2 --relative_change=1
EOT
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
python -m rl_agents.baseline_model --env_name="$name" --speed_scale=1--radars=2 --baseline_model_type="simple"
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
python -m rl_agents.baseline_model --env_name="$name" --speed_scale=1--radars=2 --baseline_model_type="no_movement"
EOT
