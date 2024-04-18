#!/bin/bash
  name='a17_3_penalty_airport_cond_a8_t30_unmask_0.1'
start=0.1
end=1.0
step=0.1

# Iterate over the sequence of floating-point numbers
for b in "0.1 0.3 0.5" do
  for l in "0.01 0.1 0.5" do
    name='a18_penalty_airport_cond_a8_t30_unmask_$l'
    for j in {0..1} do
      for i in {0..1} do
        sbatch <<EOT &
        #!/bin/bash
        #SBATCH --nodes=1
        #SBATCH --gpus-per-node=1
        #SBATCH --time=12:00:0
        #SBATCH --export=ALL
        #SBATCH --output=cphil_test_'$i'_'$j'.txt
        module load anaconda3
        echo 1
        source activate pytorch_env
        python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --cdl=1 --epsilon_init=0.5 --load_model=0 --speed_scale=1 --env_name="$name" --penalize_no_movement=1 --radars=2 --agents=$j --relative_change=$i --outside_radar_value=$l --blur_sigma=$b
        EOT
      done
    done
    for i in {0..1} do
      sbatch <<EOT &
      #!/bin/bash
      #SBATCH --nodes=1
      #SBATCH --gpus-per-node=1
      #SBATCH --time=12:00:0
      #SBATCH --export=ALL
      #SBATCH --output=cphil_test_apr16_'$i'.txt
      module load anaconda3
      echo 1
      source activate pytorch_env
      python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --cdl=1 --epsilon_init=0.5 --load_model=0 --speed_scale=1 --env_name="$name" --penalize_no_movement=1 --radars=2 --agents=2 --relative_change=$i --type_of_MARL="some_shared_info" --outside_radar_value=$l --blur_sigma=$b
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
      python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --cdl=1 --epsilon_init=0.5 --load_model=0 --speed_scale=1 --env_name="$name" --penalize_no_movement=1 --radars=2 --agents=2 --relative_change=$i --type_of_MARL="some_shared_info_shared_reward" --outside_radar_value=$l --blur_sigma=$b
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
      python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --cdl=1 --epsilon_init=0.5 --load_model=0 --speed_scale=1 --env_name="$name" --penalize_no_movement=1 --radars=2 --agents=2 --relative_change=$i --type_of_MARL="shared_targets_only" --outside_radar_value=$l --blur_sigma=$b
      EOT
    done
    for j in {1..2} do
      for i in {0..1} do
        sbatch <<EOT &
        #!/bin/bash
        #SBATCH --nodes=1
        #SBATCH --gpus-per-node=1
        #SBATCH --time=3:00:0
        #SBATCH --export=ALL
        #SBATCH --output=cphil_test_'$i'_'$j'_baseline.txt
        module load anaconda3
        echo 1
        source activate pytorch_env
        python -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main --blur_radius=1 --cdl=1 --epsilon_init=0.5 --load_model=0 --speed_scale=1 --env_name="$name" --penalize_no_movement=1 --radars=$i --agents=1 --baseline=$j --outside_radar_value=$l --blur_sigma=$b
        EOT
      done
    done
  done