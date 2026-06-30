import os
import sys
import json
import torch
import shutil
import argparse
import importlib
import torch._dynamo as dynamo
from pathlib import Path
from model_collaboration.data import eval
from multiprocessing import Pool
from model_collaboration.method import distributed_generation
from visualization import multiLLM_simulation

def run_simulation():
    torch.multiprocessing.set_start_method('spawn')

    torch.set_float32_matmul_precision('high')
    dynamo.config.cache_size_limit = 1024
    dynamo.config.recompile_limit = 1024

    # record working directory
    cwd = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, help="Path to the configuration file")
    parser.add_argument("-l", "--log_dir", default="./model_collaboration/logs/", type=str, help="Where should the log go?")
    parser.add_argument("--num_datapoints", default=None, type=int)
    parser.add_argument("--pool_llm_time_limit", default=None, type=float) # args.pool_llm_time_limit
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)

    method_name = config["method"]
    task = config["task"]
    task_type = config["task_type"]
    gpu_ids = config["gpu_ids"]
    model_names = config["model_names"]
    hyperparameters = config["hyperparameters"]

    module_path = f"model_collaboration.method.{method_name}"

    # try:
    print(f"Attempting to load module: {module_path}")
    method_module = importlib.import_module(module_path)

    if hasattr(method_module, 'run_method'):
        if args.num_datapoints == None:
            result = method_module.run_method(
                task, task_type, gpu_ids, model_names, hyperparameters
            )
        else:
            result = method_module.run_method(
                task, task_type, gpu_ids, model_names, hyperparameters, num_datapoints=args.num_datapoints # add max_generation time
            )
        print(f"Method '{method_name}' executed successfully")
    else:
        raise AttributeError(f"The module '{module_path}' does not have a 'run_method' function.")
    
    # result is the average test score!

