#!/bin/bash
name = '_airport_chance_of_no_detect_8_actions_15_targets'

sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=9:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_mar27_1.txt
module load anaconda3
echo 1
source activate pytorch_env
speed_layer = 1
speed_scale = 100

python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --cdl=1 --epsilon_init=1 --epsilon_init=0.5 --load_model=0 --env_name=$name --speed_layer=$speed_layer --speed_scale=$speed_scale
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --use_double=0 --use_dueling=0 --use_noisy=0 --use_per=0 --use_n_steps=1 --cdl=1  --epsilon_init=0.5 --load_model=0 --env_name=$name  --speed_layer=1 --speed_scale=100
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --use_double=0 --use_dueling=1 --use_noisy=0 --use_per=1 --use_n_steps=0 --cdl=1  --epsilon_init=0.5 --load_model=0 --env_name=$name  --speed_layer=1 --speed_scale=100
python -m rl_agents.baseline_model --cdl=1 --env_name=$name --speed_scale=$speed_scale
EOT
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=9:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_mar27_1.txt
module load anaconda3
echo 1
source activate pytorch_env
speed_layer = 0
speed_scale = 100

python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --cdl=1 --epsilon_init=1 --epsilon_init=0.5 --load_model=0 --env_name=$name --speed_layer=$speed_layer --speed_scale=$speed_scale
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --use_double=0 --use_dueling=0 --use_noisy=0 --use_per=0 --use_n_steps=1 --cdl=1  --epsilon_init=0.5 --load_model=0 --env_name=$name  --speed_layer=1 --speed_scale=100
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --use_double=0 --use_dueling=1 --use_noisy=0 --use_per=1 --use_n_steps=0 --cdl=1  --epsilon_init=0.5 --load_model=0 --env_name=$name  --speed_layer=1 --speed_scale=100
EOT
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=9:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_mar27_1.txt
module load anaconda3
echo 1
source activate pytorch_env
speed_layer = 1
speed_scale = 1

python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --cdl=1 --epsilon_init=1 --epsilon_init=0.5 --load_model=0 --env_name=$name --speed_layer=$speed_layer --speed_scale=$speed_scale
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --use_double=0 --use_dueling=0 --use_noisy=0 --use_per=0 --use_n_steps=1 --cdl=1  --epsilon_init=0.5 --load_model=0 --env_name=$name  --speed_layer=1 --speed_scale=100
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --use_double=0 --use_dueling=1 --use_noisy=0 --use_per=1 --use_n_steps=0 --cdl=1  --epsilon_init=0.5 --load_model=0 --env_name=$name  --speed_layer=1 --speed_scale=100
python -m rl_agents.baseline_model --cdl=1 --env_name=$name --speed_scale=$speed_scale
EOT
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=9:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_mar27_1.txt
module load anaconda3
echo 1
source activate pytorch_env
speed_layer = 0
speed_scale = 1

python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --cdl=1 --epsilon_init=1 --epsilon_init=0.5 --load_model=0 --env_name=$name --speed_layer=$speed_layer --speed_scale=$speed_scale
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --use_double=0 --use_dueling=0 --use_noisy=0 --use_per=0 --use_n_steps=1 --cdl=1  --epsilon_init=0.5 --load_model=0 --env_name=$name  --speed_layer=1 --speed_scale=100
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --use_double=0 --use_dueling=1 --use_noisy=0 --use_per=1 --use_n_steps=0 --cdl=1  --epsilon_init=0.5 --load_model=0 --env_name=$name  --speed_layer=1 --speed_scale=100
EOT
