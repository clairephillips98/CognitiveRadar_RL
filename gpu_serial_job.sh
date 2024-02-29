#!/bin/bash
salloc --time=0:15:00 --mem=4G --gres=gpu:1 --ntasks=1 --account=def-rsadve ./create_env.sh

