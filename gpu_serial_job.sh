#!/bin/bash
name='a17_2_penalty_airport_cond_a8_t30_unmask_0.1'
radars=2

sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr16_0.txt
module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --cdl=1 --epsilon_init=0.5 --load_model=0 --speed_scale=1 --env_name="$name" --penalize_no_movement=1 --radars=1 --agents=1 --baseline=2
EOT
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr16_1.txt
module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --cdl=1 --epsilon_init=0.5 --load_model=0 --speed_scale=1 --env_name="$name" --penalize_no_movement=1 --radars=1 --agents=1 --baseline=1
EOT
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr16_2.txt
module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --cdl=1 --epsilon_init=0.5 --load_model=0 --speed_scale=1 --env_name="$name" --penalize_no_movement=1 --radars=1 --agents=1 --relative_change=1
EOT
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr16_3.txt
module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --cdl=1 --epsilon_init=0.5 --load_model=0 --speed_scale=1 --env_name="$name" --penalize_no_movement=1 --radars=1 --agents=1 --relative_change=0
EOT
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr16_4.txt
module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --cdl=1 --epsilon_init=0.5 --load_model=0 --speed_scale=1 --env_name="$name" --penalize_no_movement=1 --radars=2 --agents=1 --baseline=1
EOT
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr16_5.txt
module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --cdl=1 --epsilon_init=0.5 --load_model=0 --speed_scale=1 --env_name="$name" --penalize_no_movement=1 --radars=2 --agents=1 --baseline=2
EOT
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr16_6.txt
module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --cdl=1 --epsilon_init=0.5 --load_model=0 --speed_scale=1 --env_name="$name" --penalize_no_movement=1 --radars=2 --agents=1 --relative_change=1
EOT
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr16_7.txt
module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --cdl=1 --epsilon_init=0.5 --load_model=0 --speed_scale=1 --env_name="$name" --penalize_no_movement=1 --radars=2 --agents=2 --relative_change=1
EOT
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr16_8.txt
module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --cdl=1 --epsilon_init=0.5 --load_model=0 --speed_scale=1 --env_name="$name" --penalize_no_movement=1 --radars=2 --agents=1 --relative_change=0
EOT
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr16_9.txt
module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --cdl=1 --epsilon_init=0.5 --load_model=0 --speed_scale=1 --env_name="$name" --penalize_no_movement=1 --radars=2 --agents=2 --relative_change=0
EOT
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr16_10.txt
module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --cdl=1 --epsilon_init=0.5 --load_model=0 --speed_scale=1 --env_name="$name" --penalize_no_movement=1 --radars=2 --agents=2 --relative_change=0 --type_of_MARL="some_shared_info"
EOT
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr16_11.txt
module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --cdl=1 --epsilon_init=0.5 --load_model=0 --speed_scale=1 --env_name="$name" --penalize_no_movement=1 --radars=2 --agents=2 --relative_change=0 --type_of_MARL="some_shared_info_shared_reward"
EOT
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr16_12.txt
module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --cdl=1 --epsilon_init=0.5 --load_model=0 --speed_scale=1 --env_name="$name" --penalize_no_movement=1 --radars=2 --agents=2 --relative_change=0 --type_of_MARL="shared_targets_only"
EOT
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr16_13.txt
module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --cdl=1 --epsilon_init=0.5 --load_model=0 --speed_scale=1 --env_name="$name" --penalize_no_movement=1 --radars=2 --agents=2 --relative_change=1 --type_of_MARL="some_shared_info"
EOT
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr16_14.txt
module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --cdl=1 --epsilon_init=0.5 --load_model=0 --speed_scale=1 --env_name="$name" --penalize_no_movement=1 --radars=2 --agents=2 --relative_change=1 --type_of_MARL="some_shared_info_shared_reward"
EOT
sbatch <<EOT &
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:0
#SBATCH --export=ALL
#SBATCH --output=cphil_test_apr16_15.txt
module load anaconda3
echo 1
source activate pytorch_env
python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --cdl=1 --epsilon_init=0.5 --load_model=0 --speed_scale=1 --env_name="$name" --penalize_no_movement=1 --radars=2 --agents=2 --relative_change=1 --type_of_MARL="shared_targets_only"
EOT