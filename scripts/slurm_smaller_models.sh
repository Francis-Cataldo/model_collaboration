#!/bin/bash

#SBATCH -p high-gpu-mem
#SBATCH --gres gpu:1
#SBATCH -c 1
#SBATCH --job-name=test
#SBATCH --output=output_test_new.txt
#SBATCH --time=24:00:00

module load container_env python3/2025.1-py312

crun -p .venv python -m src.run_simulation -c model_collaboration/smaller_models_3.json --num_datapoints 1 --pool_llm_time_limit 100
