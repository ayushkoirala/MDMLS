import pandas as pd
import torch
from transformers import (DPRContextEncoder, DPRContextEncoderTokenizer, 
                          DPRQuestionEncoder, DPRQuestionEncoderTokenizer)

import pandas as pd
import os
import re
import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import AutoTokenizer, AutoModelForTokenClassification
import time
import random
from tqdm import tqdm
import numpy as np
import logging
import argparse
from datetime import datetime
from rouge import Rouge 
import nltk
from nltk.tokenize import sent_tokenize
import torch.nn as nn

# Load models and tokenizers
# context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to('cuda')
# context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")




def save_abstract_embeddings(df, filename, batch_size=32):
    torch.cuda.empty_cache()  
    
    context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    gpu_ids = [i for i in range(torch.cuda.device_count())]
    if len(gpu_ids) > 1:
        print(f"Let's use {len(gpu_ids)} GPUs!")
        context_encoder = context_encoder.half().to('cuda')
        context_encoder = nn.DataParallel(context_encoder, device_ids=gpu_ids)
    else:
        context_encoder = context_encoder.to('cuda')
    
    abstracts = df['Abstract'].tolist()
    total_abstracts = len(abstracts)
    all_embeddings = []
    for i in tqdm(range(0, total_abstracts, batch_size)):
        batch_abstracts = abstracts[i:i+batch_size]
        abstract_tokens = context_tokenizer(batch_abstracts, return_tensors="pt", padding=True, truncation=True, max_length=512).to('cuda')
        abstract_embeddings = context_encoder(**abstract_tokens).pooler_output.detach().cpu().numpy()
        all_embeddings.extend(abstract_embeddings)
    
    df['Abstract embedding'] = list(all_embeddings)
    df['Abstract embedding'] = df['Abstract embedding'].apply(lambda x: ' '.join(map(str, x)))
    df.to_csv(filename)
    
    
def save_background_embeddings(df, filename, batch_size=32):
    torch.cuda.empty_cache()  
    
    background_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    background_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    
  
    gpu_ids = [i for i in range(torch.cuda.device_count())]    # Checking for multiple GPUs
    if len(gpu_ids) > 1:
        print(f"Let's use {len(gpu_ids)} GPUs!")
        background_encoder = background_encoder.half().to('cuda')
        background_encoder = nn.DataParallel(background_encoder, device_ids=gpu_ids)
    else:
        background_encoder = background_encoder.to('cuda')
    
    backgrounds = df['Background'].tolist()
    
    total_backgrounds = len(backgrounds)
    all_embeddings = []
    for i in tqdm(range(0, total_backgrounds, batch_size)):
        batch_backgrounds = backgrounds[i:i+batch_size]
        background_tokens = background_tokenizer(batch_backgrounds, return_tensors="pt", padding=True, truncation=True, max_length=512).to('cuda')
        background_embeddings = background_encoder(**background_tokens).pooler_output.detach().cpu().numpy()
        all_embeddings.extend(background_embeddings)
    df['Background embedding'] = list(all_embeddings)
    df['Background embedding'] = df['Background embedding'].apply(lambda x: ' '.join(map(str, x)))
    df.to_csv(filename)
    
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
def display(display_txt):
    logging.info(display_txt)
    print(display_txt)
    
def add_dot_product_scores(df, save_path,batch_size=32):
    df['Abstract embedding'] = df['Abstract embedding'].apply(lambda x: np.fromstring(x, sep=' '))
    df['Background embedding'] = df['Background embedding'].apply(lambda x: np.fromstring(x, sep=' '))
    total_rows = len(df)
    all_scores = []
    for i in tqdm(range(0, total_rows, batch_size)):
        multi_document_embeddings = torch.tensor(list(df['Abstract embedding'].iloc[i:i+batch_size])).float().to('cuda')
        background_embeddings = torch.tensor(list(df['Background embedding'].iloc[i:i+batch_size])).float().to('cuda')
        print(multi_document_embeddings.shape)
        print(background_embeddings.shape)
        retrieval_scores = torch.mm(multi_document_embeddings, background_embeddings.T)
        all_scores.extend(retrieval_scores.diag().cpu().numpy())
    
    df['dot_product_score'] = all_scores
    df.to_csv(save_path, index=False)
    return df

def select_top_k(df, save_path, k=6):
    
    top_k_df = df.groupby('ReviewID').apply(lambda x: x.nlargest(k, 'dot_product_score')).reset_index(drop=True)
    desired_output = []
    for _, row in top_k_df.iterrows():
        desired_output.append({
            'ReviewID': row['ReviewID'],
            'PMID': row['PMID'],
            'Background': row['Background'],
            'Candidate document': row['Abstract'],
            'ground_truth_summary': row['Target']
        })

    final_df = pd.DataFrame(desired_output)
    final_df.to_csv(save_path, index=False)



def main():
    parser = argparse.ArgumentParser(description="Data Preprocesing")
    parser.add_argument('--log', action='store_true',help='logging')
    parser.add_argument('--log_dir', type=str, default="preprocess", help='Logging directory')
    parser.add_argument('--full', action='store_true', help='Run on full data')
    #data_path = '/home/ayushk/work/Data-preprocessing-pipeline/Dataset'
    parser.add_argument('--data_path', type=str, default='/home/akoirala/Thesis/Dataset', help='Path to the dataset')
    parser.add_argument('--create_dataset', type=str, default='/home/akoirala/Thesis/Data-making-pipeline/main_dataset/result', help='Path to the result dataset')
    parser.add_argument('--top_k_batch_size', type=int, default=1, help='DPR batch size')
    parser.add_argument('--mask_ratio', type=list, default=[0.15,0.3,0.45], help='Path to the result dataset')
    args = parser.parse_args()
    if not args.full: args.log_dir += "_prototype"
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
    log_filename = "native_"
    log_filename += datetime.now().strftime("%Y_%m_%d-%H_%M")
    logging.basicConfig(
        filename=f'{args.log_dir}/{log_filename}.log',
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    process_main = log_process(f"Program Running", log_=args.log, print_=True)
    process = log_process(f"Loading data", log_=args.log, print_=True)
    
    df = pd.read_csv("/home/akoirala/Thesis/Data-making-pipeline/main_dataset/test-inputs-full.csv")
    save_abstract_embeddings(df, '/home/akoirala/Thesis/Data-making-pipeline/train-result/abstract_embeddings_test.csv')
    df = pd.read_csv(r'/home/akoirala/Thesis/Data-making-pipeline/train-result/abstract_embeddings_test.csv')
    save_background_embeddings(df, '/home/akoirala/Thesis/Data-making-pipeline/train-result/background_embeddings_test.csv')
    df = pd.read_csv(r'/home/akoirala/Thesis/Data-making-pipeline/train-result/background_embeddings_test.csv')
    df = add_dot_product_scores(df,'/home/akoirala/Thesis/Data-making-pipeline/train-result/top_k_score_test.csv')
    top_6_docs = select_top_k(df,'/home/akoirala/Thesis/Data-making-pipeline/dataset/top_6_test.csv')
   
    #save_background_embeddings(df,'abstract__background_embeddings.csv')
    #save_abstract_embeddings(df, 'abstract_embeddings.csv')



    process.end()
    process_main.end()
    

if __name__ == "__main__":
    main()
