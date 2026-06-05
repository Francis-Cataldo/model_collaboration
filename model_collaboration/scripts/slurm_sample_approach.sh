#!/bin/bash

#SBATCH -p high-gpu-mem
#SBATCH --gres gpu:1
#SBATCH -c 4
#SBATCH --job-name=test
#SBATCH --output=output.txt

module load container_env python3/2025.1-py312

crun -p .venv python model_collaboration/main.py -c model_collaboration/test_config.json
