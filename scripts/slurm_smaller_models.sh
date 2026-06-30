#!/bin/bash

#SBATCH -p high-gpu-mem
#SBATCH --gres gpu:1
#SBATCH -c 1
#SBATCH --job-name=test
#SBATCH --output=output_test_new.txt
#SBATCH --time=24:00:00

module load container_env python3/2025.1-py312

crun -p .venv python -m src.run_simulations -c model_collaboration/smaller_models.json
