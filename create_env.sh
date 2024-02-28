#!/bin/bash

# Check if the environment is "cedar"
if [ "$1" = "cedar" ]; then
    module load StdEnv/2020
fi

# Specify Python 3.11

# Create the virtual environment
python3.11 -m venv myenv

# Activate the virtual environment
source myenv/bin/activate

pip install -r requirements.txt

# Verify the Python version
python --version