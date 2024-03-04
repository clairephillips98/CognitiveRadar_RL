#!/bin/bash

echo pwd
chmod +x create_env.sh

salloc --time=2:00:00 --mem=2G --gpus-per-node=1 --ntasks=1 --mail-user=clairevphillips98@gmail.com create_env.sh cedar

