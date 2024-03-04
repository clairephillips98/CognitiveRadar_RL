#!/bin/bash

echo pwd
chmod +x create_env.sh
salloc --time -m=02:00:00 --mem=4G --gpus-per-node=1 --ntasks=1 create_env.sh cedar

