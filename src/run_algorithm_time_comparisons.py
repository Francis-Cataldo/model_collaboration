

import os
import sys
import json
import torch
import shutil
import argparse
import importlib
import torch._dynamo as dynamo
from pathlib import Path
# from model_collaboration.data import eval
from multiprocessing import Pool
# from model_collaboration.method import distributed_generation
# from visualization import multiLLM_simulation
from src.total_latency import get_total_latency_ms, get_random_total_latency_ms, ground_user_position, get_total_latency_ms_fancy
import csv
import pandas as pd
import random

latitude = random.uniform(-70.0, 70.0)
longitude = random.uniform(-180.0, 180.0)
user_pos = ground_user_position(latitude, longitude)


import time
# start_time = time.time()
total_time = 0
for i in range(1000):
    latitude = random.uniform(-70.0, 70.0)
    longitude = random.uniform(-180.0, 180.0)
    total_time += (get_total_latency_ms_fancy(user_pos, g=100, pool_llm_max_time=100, new_L=3, new_alpha=27))

print(f"fancy approach time 1000: {total_time}")

start_time = time.time()
total_time = 0
for i in range(1000):
    latitude = random.uniform(-70.0, 70.0)
    longitude = random.uniform(-180.0, 180.0)
    total_time += (get_total_latency_ms(user_pos, g=100, pool_llm_max_time=100, new_L=3, new_alpha=27))

print(f"simple approach time 1000: {time.time()-start_time}")

# print(get_random_total_latency_ms(user_pos, g=100, pool_llm_max_time=100, new_L=3, new_alpha=27))

