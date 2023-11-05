
import os
import sys
import argparse
import logging
import time
from datetime import datetime

sys.path.append('../')
#import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BartTokenizer, BartForConditionalGeneration
from rouge import Rouge
from rouge_score import rouge_scorer
from bert_score import score as bert_score
#from bart_score import BARTScorer
from transformers import AutoTokenizer, PegasusXForConditionalGeneration
from py3nvml.py3nvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from torch import nn




MODEL_PATH = {
    'bart_base':"facebook/bart-base",
    'bart_large':"facebook/bart-large",
    'bart_cnn': "facebook/bart-large-cnn",
    'bart_xsum':"facebook/bart-large-xsum"
    
    
}

class MyDataset(Dataset):
    def __init__(self, data, tokenizer,types):
        self.data = data
        if types == 'top_k_sentences':
            combined_sentences = [ sentence for sentence in data['top_k_sentences']]
            self.inputs = tokenizer(combined_sentences, padding='max_length', truncation=True, return_tensors='pt')
            self.targets = tokenizer(list(data['ground_truth_summary']), padding='max_length', truncation=True, return_tensors='pt')
        
        elif types == 'masking_background':
            combined_sentences = ['<bg>' + bg.replace("[MASK]", "<mask>") + '</bg><sent>' + sentence + '</sent>' for bg, sentence in zip(data['Masked Background'], data['top_k_sentences'])]
            self.inputs = tokenizer(combined_sentences, padding='max_length', truncation=True, return_tensors='pt')
            self.targets = tokenizer(list(data['ground_truth_summary']), padding='max_length', truncation=True, return_tensors='pt')
        
        elif types == 'bg_top_k_sentences':
            combined_sentences = ['<bg>' + bg + '</bg><sent>' + sentence + '</sent>' for bg, sentence in zip(data['Background'], data['top_k_sentences'])]
            self.inputs = tokenizer(combined_sentences, padding='max_length', truncation=True, return_tensors='pt')
            self.targets = tokenizer(list(data['ground_truth_summary']), padding='max_length', truncation=True, return_tensors='pt')
   

    def __getitem__(self, index):
        items = {
            'input_ids': self.inputs['input_ids'][index],
            'attention_mask': self.inputs['attention_mask'][index],
            'labels': self.targets['input_ids'][index],
           
        }
        return items

    def __len__(self):
        return len(self.data)


def build_loader(data,type_, split, tokenizer, bs):
    build_process = log_process(f"Buliding {split} dataset", log_=True, print_=True)
    # Build dataset
    process = log_process(f"Buliding MyDataset", log_=True, print_=True)
    dataset = MyDataset(data, tokenizer,type_)
    process.end()
    # Bulid dataloader
    process = log_process(f"Buliding DataLoader", log_=True, print_=True)
    data_loader = DataLoader(dataset, batch_size=bs, shuffle=(split=='train'))
    process.end()
    build_process.end()
    return data_loader

def train(args, train_data, val_data,type_):
    tokenizer = BartTokenizer.from_pretrained(MODEL_PATH[args.model])
    if type_ == 'masking_background' or 'bg_top_k_sentences':
        special_tokens_dict = {'additional_special_tokens': ['<bg>', '</bg>', '<sent>', '</sent>']}
        tokenizer.add_special_tokens(special_tokens_dict)
        
    model = BartForConditionalGeneration.from_pretrained(MODEL_PATH[args.model])
    model.resize_token_embeddings(len(tokenizer))
    train_loader = build_loader(train_data,type_, split='train', tokenizer=tokenizer, bs=args.bs)
    val_loader = build_loader(val_data, type_,split='val', tokenizer=tokenizer, bs=args.bs)
    
    display_cuda_info()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    
    print(torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    model.to(device)
    train_records = recorder(main_folder="random_0.15/generator_records", key_dir=args.model,types=type_)
    model_saving = save_model(model_name=args.model,type_ = type_)
    best_score = 0
    
    for epoch in range(args.n_epoch):
        train_losses = []
        model.train()
        for iter, batch in enumerate(tqdm(train_loader, desc=f"[Epoch {epoch+1}] Fine-tuning")):
            optimizer.zero_grad()
            for key in batch.keys(): 
                batch[key] = batch[key].to(device)
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if iter == 0: 
                display_cuda_info(short=True, isPrint=False)
        
        train_records.record('train_losses', train_losses)
        train_records.save()
        
        with torch.no_grad():
            gen_seq_list, ref_seq_list = [], []
            for batch in tqdm(val_loader, desc=f"[Epoch {epoch+1}] Validating"):
                for key in batch.keys(): 
                    batch[key] = batch[key].to(device)
                generated_ids = model.module.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    num_beams=args.num_beams,
                    length_penalty=2.0,
                    max_length=256,
                    min_length=32
                )
                gen_seq_list.extend([tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids])
                ref_seq_list.extend([tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['labels']])
            
            val_scores = cal_matrix(pred=gen_seq_list, ref=ref_seq_list, device=device)
            for key, val in val_scores.items(): 
                train_records.record(key, val)
            
            if val_scores['rouge1'] > best_score:
                model_saving.save(model)
                display(f"[End Epoch {epoch+1}] Reach new highest val rouge-1 ({val_scores['rouge1']*100:.2f})")
                best_score = val_scores['rouge1']
            else:
                display(f"[End Epoch {epoch+1}] Not saved model (Val rouge-1 = {val_scores['rouge1']*100:.2f})")
        
        train_records.save()




# class save_model:
#     def __init__(self,model_name, main_folder="trail/generator_checkpoints",types, subfolder='checkpoint', train_id=None):
#         self.model_name = model_name
#         self.types = types
#         if train_id:
#             self.save_dir = f"{main_folder}/{self.model_name}{subfolder}/{train_id}"
#             if not os.path.exists(self.save_dir): 
#                 os.makedirs(self.save_dir)
#         else:
#             model_folder = os.path.join(main_folder, self.model_name)
#             if not os.path.exists(model_folder): 
#                 os.makedirs(model_folder)
#                 max_checkpoint = 0
#             else:
#                 existing_folders = [folder for folder in os.listdir(model_folder) if folder.startswith(f'{subfolder}-')]
#                 max_checkpoint = max([int(folder.split('-')[1]) for folder in existing_folders], default=0)
#             self.save_dir = f"{model_folder}/{subfolder}-{max_checkpoint+1}"

#     def save(self, model):
#         if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
#         model.module.save_pretrained(self.save_dir)
#         display(f"** Saved model to {self.save_dir}")

class save_model:
    def __init__(self, model_name,  type_,main_folder="random_0.15/generator_checkpoints", subfolder='checkpoint', train_id=None):
        self.model_name = model_name
        self.types = type_
        if train_id:
            # Include 'types' in the save directory
            self.save_dir = f"{main_folder}/{self.model_name}/{self.types}/{subfolder}/{train_id}"
            if not os.path.exists(self.save_dir): 
                os.makedirs(self.save_dir)
        else:
            model_folder = os.path.join(main_folder, self.model_name, self.types)
            if not os.path.exists(model_folder): 
                os.makedirs(model_folder)
                max_checkpoint = 0
            else:
                existing_folders = [folder for folder in os.listdir(model_folder) if folder.startswith(f'{subfolder}-')]
                max_checkpoint = max([int(folder.split('-')[1]) for folder in existing_folders], default=0)
            # Include 'types' in the save directory
            self.save_dir = f"{model_folder}/{subfolder}-{max_checkpoint+1}"

    def save(self, model):
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        model.module.save_pretrained(self.save_dir)
        display(f"** Saved model to {self.save_dir}")  

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

def cal_bertscore(pred, ref, mode='avg'):
    precision, recall, f1 = bert_score(pred, ref, lang="en", verbose=True)
    if mode=='avg':
        return f1.mean().item()
    else:
        # return f1
        return [score.item() for score in f1]


def cal_rouge(hyps, refs, mode='avg'):
    metrics = ['rouge1', 'rouge2', 'rougeL']
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
    if len(refs)!=len(hyps):
        raise Exception(f"No. of Refs and Hyps are not equal")
    results = {}
    for metric in metrics: results[metric]=[] 
    for idx in range(len(refs)):
        scores = scorer.score(refs[idx].strip(), hyps[idx].strip())
        for metric in metrics: results[metric].append(scores[metric].fmeasure)
    if mode=='avg':
        results = {rouge_metric: np.average(rouge_metric_scores) for (rouge_metric, rouge_metric_scores) in results.items()}
    # else:
    #     results = {rouge_metric: rouge_metric_scores for (rouge_metric, rouge_metric_scores) in results.items()}
    return results

# def cal_bartscore(pred, ref, mode='avg', device='cpu'):
#     bart_scorer = BARTScorer(device=device, checkpoint=MODEL_PATH[args.model])
#     scores = bart_scorer.score(pred, ref)
#     if mode=='avg':
#         return sum(scores)/len(scores)
#     else:
#         return scores

def cal_matrix(pred, ref, mode='avg', device='cpu', metrix=['rouge', 'bertscore', 'bartscore']):
    process = log_process('Calculating Rouge score', print_=True)
    if 'rouge' in metrix: eval_score = cal_rouge(pred, ref, mode)
    process.end()
    process = log_process('Calculating BERT score', print_=True)
    process.end()
    if 'bertscore' in metrix: eval_score['BERTscore'] = cal_bertscore(pred, ref, mode)
    #process = log_process('Calculating BART score', print_=True)
    #if 'bartscore' in metrix: eval_score['BARTscore'] = cal_bartscore(pred, ref, mode, device)
    process.end()
    return eval_score

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

import os
import numpy as np
import json
from datetime import datetime

# class recorder:
#     def __init__(self, main_folder="records", key_dir="default", types=''):
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         self.save_dir = os.path.join(main_folder, key_dir, types)
#         self.record_dict = {}

#     def record(self, key, val):
#         """Record the values under the given key. If the key exists, append to it. If not, create a new list."""
#         self.record_dict.setdefault(key, []).append(val)

#     def save(self):
#         """Save the records to the appropriate format."""
#         if not os.path.exists(self.save_dir): 
#             os.makedirs(self.save_dir)
            
#         for key, val in self.record_dict.items():
#             if isinstance(val[0], (int, float, str, dict, list)):
#                 # Save as JSON
#                 filepath = os.path.join(self.save_dir, f"{key}.json")
#                 with open(filepath, 'w') as json_file:
#                     json.dump(val, json_file)
#                 display(f"Saved {key} to {filepath}")
#             else:
#                 # Save as NPY
#                 filepath = os.path.join(self.save_dir, f"{key}.npy")
#                 np.save(filepath, np.array(val))
#                 display(f"Saved {key} to {filepath}")


class recorder:
    def __init__(self, main_folder="records", key_dir="default", types='default'):
        self.save_dir = os.path.join(main_folder, key_dir,types)
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

