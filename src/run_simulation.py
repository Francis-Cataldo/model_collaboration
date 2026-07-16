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
from src.total_latency import get_total_latency_ms, get_random_total_latency_ms, ground_user_position
import csv
import pandas as pd
import random

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
    parser.add_argument("--g", type=float, default=1)
    parser.add_argument("--L", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=None)
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

    # want time for total system 

    # get timing!
    # for _ in range(args.num_datapoints):
    #     latitude = random.uniform(-70.0, 70.0)
    #     longitude = random.uniform(-180.0, 180.0)
    #     user_pos = ground_user_position(latitude, longitude)
        
        
    #     time = get_total_latency_ms(user_pos, pool_llm_max_time=args.pool_llm_time_limit, g=args.g, new_L=int(args.L)) # this is for one prompt
    #     random_time = get_random_total_latency_ms(user_pos, pool_llm_max_time=args.pool_llm_time_limit, g=args.g, new_L=int(args.L))
    #     print(time)
    #     with open("final_timing_data.csv", "a", newline="\n") as file:
    #         writer = csv.writer(file)
    #         writer.writerow([args.L, time, random_time])

    
    # df.to_csv('data/dataframe_output.csv', mode="a", index=False)

    if hasattr(method_module, 'run_method'):
        if args.num_datapoints == None:
            result = method_module.run_method(
                task, task_type, gpu_ids, model_names, hyperparameters, args.g, args.L, pool_llm_time_limit=args.pool_llm_time_limit, alpha=args.alpha
            )
        else:
            result = method_module.run_method(
                task, task_type, gpu_ids, model_names, hyperparameters, args.g, args.L, num_datapoints=args.num_datapoints, pool_llm_time_limit=args.pool_llm_time_limit, alpha=args.alpha # add max_generation time
            )
        print(f"Method '{method_name}' executed successfully")
    else:
        raise AttributeError(f"The module '{module_path}' does not have a 'run_method' function.")
    
    # # result is the average test score!

    
    # # time = get_total_latency_ms(pool_llm_max_time=5, g=args.g) # this is for one prompt

    with open("final_results_data.csv", "a", newline="\n") as file:
        writer = csv.writer(file)
        writer.writerow([args.L, result])

    return None


    # want a csv with datapoints of system time and response quality (measured by performance on metrics)

run_simulation()


