import os
import csv
import sys
import time
import json
import logging
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import mean_squared_error



def display(display_txt):
    logging.info(display_txt)
    print(display_txt)

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
        elapse_min = int(elapsed_time / 60)
        self.display(f"End '{self.process_name}' (in {elapse_min}:{elapse_sec} mins)")
    
    def display(self, display_txt):
        if self.log_: logging.info(display_txt)
        if self.print_: print(display_txt)

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


class CustomDataset(Dataset):
    def __init__(self, inputs, target):
        self.inputs = inputs
        self.target = target
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.inputs.items()}
        item['labels'] = self.target[idx]
        return item
    
    def __len__(self):
        return len(self.target)



def train_and_validate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
    
    train_df = pd.read_csv(args.train_file)
    val_df = pd.read_csv(args.val_file)
    
    train_inputs = tokenizer(list(train_df['input']), padding="max_length", truncation=True, return_tensors="pt")
    val_inputs = tokenizer(list(val_df['input']), padding="max_length", truncation=True, return_tensors="pt")
    
    train_dataset = CustomDataset(train_inputs, torch.tensor(train_df['final_score']))
    val_dataset = CustomDataset(val_inputs, torch.tensor(val_df['final_score']))
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.MSELoss()
    model.to(device)
    
    records = recorder()
    
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        
        for batch in tqdm(train_dataloader, desc=f"[Epoch {epoch+1}] Training"):
            optimizer.zero_grad()
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            
            outputs = model(**batch)
            loss = loss_fn(outputs.logits.squeeze(), batch['labels'].float())
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        display(f"[Epoch {epoch+1}] Training loss: {avg_train_loss:.4f}")
        records.record('train_losses', avg_train_loss)
        
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"[Epoch {epoch+1}] Validating"):
                for key in batch.keys():
                    batch[key] = batch[key].to(device)
                
                outputs = model(**batch)
                loss = loss_fn(outputs.logits.squeeze(), batch['labels'].float())
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        display(f"[Epoch {epoch+1}] Validation loss: {avg_val_loss:.4f}")
        records.record('val_losses', avg_val_loss)
        records.save()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and validation')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--train_file', type=str, required=True, help='Path to the training file',default='/home/akoirala/Thesis/Query_modeling/training.csv')
    parser.add_argument('--val_file', type=str, required=True, help='Path to the validation file',default='/home/akoirala/Thesis/Query_modeling/validation.csv')
    args = parser.parse_args()
    
    logging.basicConfig(filename='training.log',
                        level=logging.INFO,
                        format='%(asctime)s [%(levelname)s]: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    
    train_and_validate(args)
