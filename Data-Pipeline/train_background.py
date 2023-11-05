import os
import sys
import argparse
import logging
from datetime import datetime

sys.path.append('../')
import common
#from common import recorder

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import mean_squared_error

import ast


# class recorder:
#     def __init__(self, main_folder="trail/records", subfolder='rec', train_id=None):
#         if train_id:
#             self.save_dir = f"{main_folder}/{subfolder}/{train_id}"
#         else:
#             if not os.path.exists(main_folder): 
#                 os.makedirs(main_folder)
#                 max_checkpoint = 0
#             else:
#                 existing_folders = [folder for folder in os.listdir(main_folder) if folder.startswith(f'{subfolder}-')]
#                 max_checkpoint = max([int(folder.split('-')[1]) for folder in existing_folders], default=0)
#             self.save_dir = f"{main_folder}/{subfolder}-{max_checkpoint+1}"
#         self.record_dict = {}

#     def record(self, key, val):
#         if key not in self.record_dict.keys(): self.record_dict[key] = [val]
#         else: self.record_dict[key].append(val)
        
#     def save(self):
#         if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
#         for key, val in self.record_dict.items():
#             filepath = f"{self.save_dir}/{key}.npy"
#             np.save(filepath, np.array(val))
#             display(f"Saved {key} to {filepath}")

# def display(display_txt):
#     logging.info(display_txt)
#     print(display_txt)
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
    def __init__(self, main_folder="random_0.15/background_checkpoints", subfolder='checkpoint', train_id=None):
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
# def save_model(args, model):
#     # If the model is of DataParallel type, access the original model object
#     if isinstance(model, torch.nn.DataParallel):
#         model = model.module
    
#     main_folder = os.path.join('/home/akoirala/Thesis/Final-pipeline/random_0.15',args.model_dir)
#     if not os.path.exists(main_folder): 
#         os.makedirs(args.model_dir)
#         max_checkpoint = 0
#     else:
#         existing_folders = [folder for folder in os.listdir(main_folder) if folder.startswith(f'{args.model_dir}-')]
#         max_checkpoint = max([int(folder.split('-')[1]) for folder in existing_folders], default=0)
    
#     checkpoint_dir = f"{main_folder}/{args.model_dir}-{max_checkpoint+1}"
#     if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
#     model.save_pretrained(checkpoint_dir)
#     torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'model_weights.pt'))
#     common.display(f"Saved model to {checkpoint_dir}")

def training(args, train_df, val_df):
    # Check if CUDA (GPU) is available, else use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
    gpu_ids = [int(id) for id in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if id.strip()]

    if len(gpu_ids) > 1:
        print(f"Let's use {len(gpu_ids)} GPUs!")
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    process = common.log_process(f"Tokenizing (Train dataset)", log_=args.log, print_=True)
    train_inputs = tokenizer(list(train_df['input_seq']), padding="max_length", truncation=True, return_tensors="pt")
    process.end()
    process = common.log_process(f"Tokenizing (Val dataset)", log_=args.log, print_=True)
    val_inputs = tokenizer(list(val_df['input_seq']), padding="max_length", truncation=True, return_tensors="pt")
    process.end()
    # Create DataLoader
    process = common.log_process(f"Building dataset", log_=args.log, print_=True)
    train_dataset = myDataset(train_inputs, torch.tensor(train_df['final_score']))
    val_dataset = myDataset(val_inputs, torch.tensor(val_df['final_score']))
    process.end()
    process = common.log_process(f"Building dataloader", log_=args.log, print_=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)
    process.end()
    # Training
    optimizer = AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    model.to(device)
    #records = recorder()
    train_records = recorder(main_folder="random_0.15/background_records")
    model_saving = save_model()
    best_loss = float('inf')
    # Display mem info before trainning
    common.display_cuda_info()
    for epoch in range(args.n_epoch):  # Number of epochs
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
        common.display(f"[Epoch {epoch+1}] Trainning loss: {avg_train_loss:.4f}")
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
        avg_val_loss = sum(val_losses)/len(val_losses)
        if avg_val_loss<best_loss:
            common.display(f"[Epoch {epoch+1}] New lowest val loss ({avg_val_loss:.4f})")
            model_saving.save(model)
            best_loss = avg_val_loss
        else:
            common.display(f"[Epoch {epoch+1}] Val loss = {avg_val_loss:.4f} (Not reach new best)")
        train_records.record('val_losses', avg_val_loss)
        train_records.save()
    

def preprocessing(csv_path):
    df = pd.read_csv(csv_path)
    return df

# def main():
#     parser = argparse.ArgumentParser(description='Extracting')
#     # Training settings
#     parser.add_argument('--bs', type=int, default=128,
#                         help='batch size (default: 16)')
#     parser.add_argument('--n_epoch', type=int, default=3,
#                         help='number of epochs (default: 5)')
#     parser.add_argument('--lr', type=float, default=3e-5,
#                         help='learning rate')
#     # Result path settings
#     parser.add_argument('--model_dir', type=str, default="checkpoint",
#                         help='saved model directory')
#     # File running settings
#     parser.add_argument('--full', action='store_true',
#                         help='run full samples')
#     parser.add_argument('--log', action='store_true',
#                         help='logging')
#     parser.add_argument('--log_dir', type=str, default="mylog",
#                         help='logging directory')
#     args = parser.parse_args()

#     if not args.full: args.log_dir += "_prototype"
#     if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
#     log_filename = "native_"
#     log_filename += datetime.now().strftime("%Y_%m_%d-%H_%M")
#     logging.basicConfig(
#         filename=f'{args.log_dir}/{log_filename}.log',
#         level=logging.INFO,
#         format="%(asctime)s [%(levelname)s]: %(message)s",
#         datefmt="%Y-%m-%d %H:%M:%S",
#     )
#     common.display(f"{args=}")
#     process_main = common.log_process(f"Main program", log_=args.log, print_=True)

#     process = common.log_process(f"Loading data", log_=args.log, print_=True)
#     train_df = preprocessing(r'/home/akoirala/Thesis/Data-Preprocessing-Pipeline/result/PICO-0.15/PICO_0.15_train.csv')
#     common.display(f"Imported {len(train_df)} rows of train datasset")
#     val_df = preprocessing(r'/home/akoirala/Thesis/Data-Preprocessing-Pipeline/result/PICO-0.15/PICO_0.15_dev.csv')
#     common.display(f"Imported {len(train_df)} rows of train datasset")
#     common.display(f"Imported {len(val_df)} rows of validation dataset")
#     process.end()

#     process = common.log_process(f"Training", log_=args.log, print_=True)
#     training(args, train_df, val_df)
#     process.end()

#     process_main.end()

# if __name__ == '__main__':
#     main()