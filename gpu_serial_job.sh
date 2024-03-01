#!/bin/bash

echo pwd

salloc --time=0:15:00 --mem=4G --gres=gpu:1 --ntasks=1 --account=def-rsadve /home/cphil/scratch/CognitiveRadar_RL/create_env.sh cedar

