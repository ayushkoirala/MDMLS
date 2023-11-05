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
    def __init__(self, main_folder="random_0.15/records", subfolder='rec', train_id=None):
        # if train_id:
        #     self.save_dir = f"{main_folder}/{subfolder}/{train_id}"
        # else:
        #     if not os.path.exists(main_folder): 
        #         os.makedirs(main_folder)
        #         max_checkpoint = 0
        #     else:
        #         existing_folders = [folder for folder in os.listdir(main_folder) if folder.startswith(f'{subfolder}-')]
        #         max_checkpoint = max([int(folder.split('-')[1]) for folder in existing_folders], default=0)
        #     self.save_dir = f"{main_folder}/{subfolder}-{max_checkpoint+1}"
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
    

# ======================================== Memmory management ========================================
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

# ======================================== Data importing ========================================
# def import_raw_mup(split):
#     filepath = f"{get_base_dir()}/mup_dataset/raw/{split}.jsonl"
#     with open(filepath, 'r') as json_file:
#         json_list = list(json_file)
#     dataset_list = []
#     data_len = len(json_list)
#     for i, json_str in enumerate(json_list):
#         data = json.loads(json_str)
#         if split=='test': summary = "<PAD>" 
#         else: summary = data["summary"]
#         dataset_list.append({
#             "paper_id": data["paper_id"], 
#             "abstract": data["paper"]["abstractText"], 
#             "sections": data["paper"]["sections"], 
#             "summary": summary
#         })
#         print_progress(i, data_len, desc=f'Loading abstract input ({split})')
#     return pd.DataFrame(dataset_list)

# def import_graph_sum(split):
#     filepath = f"{get_base_dir()}/mup_dataset/graph_sum/{split}.json"
#     with open(filepath, 'r') as json_file:
#         json_list = list(json_file)
#     return [json.loads(json_str) for json_str in json_list]

# def import_sentence_df(split, full=True):
#     sentence_dir = f"{get_base_dir()}/mup_dataset/sentences_data"
#     if not full: sentence_dir += '_prototype'
#     sentence_df = pd.read_csv(f"{sentence_dir}/dataset_{split}.csv")
#     return sentence_df

# # ======================================== Data preprocessing (Score Ranking) ========================================
# def cal_score(row, lamb=0.15):
#     f1_score = {}
#     for r in ['R1', 'R2']:
#         div = row[f'{r}_p']+row[f'{r}_r']
#         if div==0: f1_score[r]=0
#         else: f1_score[r] = 2*(row[f'{r}_p']*row[f'{r}_r']) / div
#     score = (lamb*f1_score['R1']) + f1_score['R2']
#     return score

# def mask_sum(row):
#     all_sentences = [j for i in row["sentences"] for j in i]
#     flatten_ner = [j for i in list(row["predicted_ner"]) for j in i]
#     mask_sentences = []
#     previous_end = 0
#     for ner in flatten_ner:
#         mask_sentences += all_sentences[previous_end:ner[0]]
#         mask_sentences.append("[MASK]")
#         previous_end = ner[1]+1
#     mask_sentences += all_sentences[previous_end:]
#     return " ".join(mask_sentences)

# def build_input(row):
#     return row["UMR"] + " [SEP] " + row["doc_sentence"]
#     # return "[CLS] " + row["UMR"] + " [SEP] " + row["doc_sentence"] + " [SEP]"

# # def count_ent(text):
# #     doc = nlp(text)
# #     return len(doc)

# def preprocessing(args, split):
#     data_dir = f"{get_base_dir()}/mup_dataset/sentences_data"
#     if not args.full: data_dir += "_prototype"
#     # Sentence df
#     sentence_df = pd.read_csv(f"{data_dir}/dataset_{split}.csv")
#     sent_data_len = len(sentence_df)
#     print(f"Imported {sent_data_len} rows of sentence dataset")
#     sentence_df.dropna(inplace=True)
#     print(f"Droped {sent_data_len-len(sentence_df)} rows of None score (Remain: {len(sentence_df)})")
#     if args.filter_sent:
#         tqdm.pandas(desc="Filtering sentence")
#         # mask = (sentence_df['doc_sentence'].progress_apply(count_ent))>=3
#         mask = (sentence_df['doc_sentence'].progress_apply(len))>=10
#         sent_data_len = len(sentence_df)
#         sentence_df = sentence_df[mask]
#         print(f"Filtered out {sent_data_len-len(sentence_df)} rows (Remain: {len(sentence_df)})")
#     tqdm.pandas(desc="calculating score")
#     sentence_df['score'] = sentence_df.progress_apply(cal_score, axis=1)
#     # Summary dfa
#     summary_df = pd.read_csv(f"{data_dir}/summary_{split}.csv")
#     print(f"Imported {len(summary_df)} rows of summary dataset")
#     for col in ['sentences', 'predicted_ner', 'predicted_re']:
#         summary_df[col] = summary_df[col].apply(ast.literal_eval)
#     tqdm.pandas(desc="UMR processing")
#     summary_df['UMR'] = summary_df.progress_apply(mask_sum, axis=1)
#     # Merge & Build datset
#     merge_df = pd.merge(sentence_df, summary_df, on=['paper_id', 'summary_id'], how='inner')
#     print(f"Merged to {sent_data_len} rows")
#     dataset_df = merge_df.loc[:, ["paper_id", "summary_id", "sentence_id"]]
#     tqdm.pandas(desc="building input seq")
#     dataset_df["input"] = merge_df.progress_apply(build_input, axis=1)
#     dataset_df["target"] = merge_df.loc[:, 'score']
#     return dataset_df

# def sampling_out(dataset_df, threashold_score=0.05):
#     # low samples
#     mask = dataset_df['target']<threashold_score
#     low_samples = dataset_df[mask]
#     random_sample = dataset_df[mask].sample(n=int(len(low_samples)*0.1), replace=False)
#     # high samples
#     mask = dataset_df['target']>=threashold_score
#     high_samples = dataset_df[mask]
#     df_result = pd.concat([high_samples,random_sample], ignore_index=False).sort_index()
#     return df_result

# class Args(argparse.Namespace):
#     pass

# # ======================================== Graph visualization ========================================
# # import spacy
# # from spacy import displacy
# # from spacy.tokens import Span
# # from pathlib import Path

# # def display_ent(data, savefile=None):
# #     all_sentences = [j for i in data["sentences"] for j in i]
# #     flatten_ner = [j for i in data["predicted_ner"] for j in i]

# #     colors = {
# #         'Task':     "#DE3163", 
# #         "Method":   "#6495ED", 
# #         "Metric":   "#FF00FF", 
# #         "Material": "#40E0D0", 
# #         "OtherScientificTerm": "#9FE2BF", 
# #         "Generic":  "#FFBF00",
# #     }
# #     options = {"ents": colors.keys(), "colors": colors, "compact": True,}
# #     nlp = spacy.load("en_core_web_sm")
# #     doc = nlp(" ".join(all_sentences))
# #     span_list = [Span(doc, start, end+1, ent_type) for start, end, ent_type in flatten_ner]
# #     doc.set_ents(span_list)
# #     ents = list(doc.ents)
# #     result = displacy.render(doc, style="ent", options=options)
# #     if not (savefile is None):
# #         output_path = Path("_output_imgs/"+savefile)
# #         output_path.open("w", encoding="utf-8").write(result)
# # =====================================================================================================