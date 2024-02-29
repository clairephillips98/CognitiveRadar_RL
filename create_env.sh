#!/bin/bash

# Check if the environment is "cedar"
if [ "$1" = "cedar" ]; then
    module load StdEnv/2020 python/3.11
fi

# Specify Python 3.11

# Create the virtual environment
python3 -m venv myenv

# Activate the virtual environment
source myenv/bin/activate

pip3 install -r requirements.txt

# Verify the Python version

python --version

python3 -m rl_agents.DRL_code_pytorch.Rainbow_DQN.Rainbow_DQN_radar_main