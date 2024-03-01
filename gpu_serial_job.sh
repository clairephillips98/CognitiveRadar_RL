#!/bin/bash

echo pwd
chmod +x create_env.sh

salloc --time=2:00:00 --mem=4G --gres=gpu:1 --ntasks=1 create_env.sh cedar

