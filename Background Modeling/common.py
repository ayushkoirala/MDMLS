import os
import csv
import sys
import time
import json
import logging
import argparse

import pandas as pd
import torch
from pynvml import *
import ast
from tqdm import tqdm
import numpy as np

def print_progress(curr, full, desc='', bar_size=30):    
    bar = int((curr+1)/full*bar_size)
    sys.stdout.write(f"\r{desc}[{'='*bar}{' '*(bar_size-bar)}] {curr+1}/{full}")
    if curr+1==full: print()

class log_process():
    def __init__(self, process_name, log_=True, print_=False):
        self.process_name = process_name
        self.start_time = time.time()
        self.log_ = log_
        self.print_ = print_
        self.display(f"Starting '{self.process_name}'")

    def end(self):
        elapsed_time = time.time() - self.start_time
        elapse_sec = int(elapsed_time % 60)
        if elapse_sec<10: elapse_sec = '0'+str(elapse_sec)
        elapse_min = int(elapsed_time/60)
        if elapse_min<60:
            self.display(f"End '{self.process_name}' (in {elapse_min}:{elapse_sec} mins)")
        else:
            elapse_hr = elapse_min//60
            elapse_min = elapse_min%60
            if elapse_min<10: elapse_min = '0'+str(elapse_min)
            self.display(f"End '{self.process_name}' (in {elapse_hr}:{elapse_min}:{elapse_sec} hrs)")

    def display(self, display_txt):
        if self.log_: logging.info(display_txt)
        if self.print_: print(display_txt)

class recorder:
    def __init__(self, main_folder="records", subfolder='rec', train_id=None):
        if train_id:
            self.save_dir = f"{main_folder}/{subfolder}/{train_id}"
        else:
            if not os.path.exists(main_folder): 
                os.makedirs(main_folder)
                max_checkpoint = 0
            else:
                existing_folders = [folder for folder in os.listdir(main_folder) if folder.startswith(f'{subfolder}-')]
                max_checkpoint = max([int(folder.split('-')[1]) for folder in existing_folders], default=0)
            self.save_dir = f"{main_folder}/{subfolder}-{max_checkpoint+1}"
        self.record_dict = {}

    def record(self, key, val):
        if key not in self.record_dict.keys(): self.record_dict[key] = [val]
        else: self.record_dict[key].append(val)
        
    def save(self):
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        for key, val in self.record_dict.items():
            filepath = f"{self.save_dir}/{key}.npy"
            np.save(filepath, np.array(val))
            display(f"Saved {key} to {filepath}")

    
def get_base_dir():
    current_dir = os.getcwd().split("/")
    index = current_dir.index("Query_based_model")+1
    base_dir = '/'.join(current_dir[:index])
    return base_dir


def display(display_txt):
    logging.info(display_txt)
    print(display_txt)


def get_cuda_info():
    nvmlInit()
    num_gpus = torch.cuda.device_count()
    gpus_mem = {}
    for n in range(num_gpus):
        h = nvmlDeviceGetHandleByIndex(n)
        info = nvmlDeviceGetMemoryInfo(h)
        gpus_mem[n] = {}
        gpus_mem[n]['gpu_name'] = torch.cuda.get_device_name(n)
        gpus_mem[n]['mem_total'] = info.total/(1024**3)
        gpus_mem[n]['mem_free'] = info.free/(1024**3)
        gpus_mem[n]['mem_used'] = info.used/(1024**3)
    return gpus_mem

def display_cuda_info(short=False, isPrint=True, isLog=True):
    cuda_info_list = get_cuda_info()
    if short:
        text = " | ".join([f"D{idx}: {cuda_info['mem_used']:.2f}/{cuda_info['mem_total']:.2f} GB" for idx, cuda_info in enumerate(cuda_info_list.values())])
        if isPrint: print(text)
        if isLog: logging.info(text)
    else:
        for idx, cuda_info in enumerate(cuda_info_list.values()):
            text = f"""Device {idx}: {cuda_info['gpu_name']} | Used: {cuda_info['mem_used']:.2f}/{cuda_info['mem_total']:.2f} GB"""
            if isPrint: print(text)
            if isLog: logging.info(text)
