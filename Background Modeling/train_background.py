import os
import sys
import argparse
import logging
from datetime import datetime
import time
sys.path.append('../')
#import common
#from common import recorder

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import mean_squared_error
from datetime import datetime
import ast
from torch.utils.data import DataLoader, TensorDataset

def create_dataloader(df, inputs, batch_size):
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], torch.tensor(df['final_score']))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader
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
def display(display_txt):
    logging.info(display_txt)
    print(display_txt)
def display(display_txt):
    logging.info(display_txt)
    print(display_txt)
class recorder:
    def __init__(self, main_folder="records"):
        self.save_dir = main_folder
        self.record_dict = {}

    def record(self, key, val):
        if key not in self.record_dict.keys(): 
            self.record_dict[key] = [val]
        else: 
            self.record_dict[key].append(val)

    def save(self):
        if not os.path.exists(self.save_dir): 
            os.makedirs(self.save_dir)
        for key, val in self.record_dict.items():
            filepath = os.path.join(self.save_dir, f"{key}.npy")
            np.save(filepath, np.array(val))
            display(f"Saved {key} to {filepath}")
    
class myDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, target):
        self.inputs = inputs
        self.target = target

    def __getitem__(self, idx):
        item = {key: val[idx] for key,val in self.inputs.items()}
        item['labels'] = self.target[idx]
        return item

    def __len__(self):
        return len(self.target)
    
class save_model:
    def __init__(self, main_folder="pico_0.3/background_checkpoints", subfolder='checkpoint', train_id=None):
        if train_id:
            self.save_dir = f"{main_folder}/{subfolder}/{train_id}"
            if not os.path.exists(self.save_dir): 
                os.makedirs(self.save_dir)
        else:
            model_folder = main_folder
            if not os.path.exists(model_folder): 
                os.makedirs(model_folder)
                max_checkpoint = 0
            else:
                existing_folders = [folder for folder in os.listdir(model_folder) if folder.startswith(f'{subfolder}-')]
                max_checkpoint = max([int(folder.split('-')[1]) for folder in existing_folders], default=0)
            self.save_dir = f"{model_folder}/{subfolder}-{max_checkpoint+1}"

    def save(self, model):
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        model.module.save_pretrained(self.save_dir)
        display(f"** Saved model to {self.save_dir}")

def training(args, train_df, val_df):
    # Check if CUDA (GPU) is available, else use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
    gpu_ids = [int(id) for id in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if id.strip()]
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    # if len(gpu_ids) > 1:
    #     print(f"Let's use {len(gpu_ids)} GPUs!")
    #     model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    process = log_process(f"Tokenizing (Train dataset)", log_=args.log, print_=True)
    train_inputs = tokenizer(list(train_df['input_seq']), padding="max_length", truncation=True, return_tensors="pt")
    process.end()
    process = log_process(f"Tokenizing (Val dataset)", log_=args.log, print_=True)
    val_inputs = tokenizer(list(val_df['input_seq']), padding="max_length", truncation=True, return_tensors="pt")
    process.end()
    # Create DataLoader
    process = log_process(f"Building dataset", log_=args.log, print_=True)
    train_dataset = myDataset(train_inputs, torch.tensor(train_df['final_score']))
    val_dataset = myDataset(val_inputs, torch.tensor(val_df['final_score']))
    process.end()
    process = log_process(f"Building dataloader", log_=args.log, print_=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)
    process.end()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    model.to(device)
    #records = recorder()
    train_records = recorder(main_folder="pico_0.3/background_records")
    model_saving = save_model()
    best_loss = float('inf')
    # Display mem info before trainning
    #display_cuda_info()
    for epoch in range(args.n_epoch):  # Number of epochs
        start_time = datetime.now()  
        model.train()
        train_losses = []
        # Train
        for batch in tqdm(train_dataloader, desc=f"[Epoch {epoch+1}] Training"):
            optimizer.zero_grad()
            for key in batch.keys(): batch[key] = batch[key].to(device)
            outputs = model(batch['input_ids'], attention_mask=batch['attention_mask'])
            loss = loss_fn(outputs.logits.squeeze(), batch['labels'].float())  # Assuming 1D output
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_train_loss = sum(train_losses)/len(train_losses)
        display(f"[Epoch {epoch+1}] Trainning loss: {avg_train_loss:.4f}")
        train_records.record('train_losses', train_losses)
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"[Epoch {epoch+1}] Validating"):
                for key in batch.keys(): batch[key] = batch[key].to(device)
                outputs = model(batch['input_ids'], attention_mask=batch['attention_mask'])
                loss = loss_fn(outputs.logits.squeeze(), batch['labels'].float())  # Assuming 1D output
                val_losses.append(loss.item())
        end_time = datetime.now() 
        epoch_duration = end_time - start_time
        display(f"[Epoch {epoch+1}] Time taken: {epoch_duration}")
        avg_val_loss = sum(val_losses)/len(val_losses)
        if avg_val_loss<best_loss:
            display(f"[Epoch {epoch+1}] New lowest val loss ({avg_val_loss:.4f})")
            model_saving.save(model)
            best_loss = avg_val_loss
        else:
            display(f"[Epoch {epoch+1}] Val loss = {avg_val_loss:.4f} (Not reach new best)")
        train_records.record('val_losses', avg_val_loss)
        train_records.save()
    

def preprocessing(csv_path):
    df = pd.read_csv(csv_path)
    return df
