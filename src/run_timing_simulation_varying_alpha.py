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
from src.total_latency import get_total_latency_ms, get_random_total_latency_ms, get_total_latency_ms_closest, ground_user_position
import csv
import pandas as pd
import random

ranking_time = 1
fuser_time = 5


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
alpha = args.alpha/5.0

simulations_per_datapoint = 1000
for i in range(simulations_per_datapoint):

    latitude = random.uniform(-70.0, 70.0)
    longitude = random.uniform(-180.0, 180.0)
    user_pos = ground_user_position(latitude, longitude)
    
    
    time_exp = get_total_latency_ms(user_pos, pool_llm_max_time=args.pool_llm_time_limit, g=args.g, new_L=int(args.L), new_alpha=alpha, ranker_time=ranking_time, fuser_time=fuser_time) # this is for one prompt
    random_time = get_random_total_latency_ms(user_pos, pool_llm_max_time=args.pool_llm_time_limit, g=args.g, new_L=int(args.L), new_alpha=alpha, ranker_time=ranking_time, fuser_time=fuser_time)
    closest_time = get_total_latency_ms_closest(user_pos, pool_llm_max_time=args.pool_llm_time_limit, g=args.g, new_L=int(args.L), new_alpha=alpha, ranker_time=ranking_time, fuser_time=fuser_time)


    with open(f"final_timing_data_varying_alpha_{alpha}.csv", "a", newline="\n") as file:
        writer = csv.writer(file)
        writer.writerow([alpha, time_exp, random_time, closest_time])