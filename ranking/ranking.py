import ast
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset
from rouge import Rouge
import os
import logging
import time
import seaborn as sns
from datetime import datetime

def load_data(path):
    return pd.read_csv(path)

def init_model_and_tokenizer(checkpoint):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=1)
    return tokenizer, model

def create_dataloader(df, tokenizer, batch_size):
    df.dropna(subset=['input_seq'])
    df = df[df['input_seq'].apply(lambda x: isinstance(x, str))]
    inputs = tokenizer(list(df['input_seq']), padding="max_length", truncation=True, return_tensors="pt")
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], torch.tensor(df['final_score']))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def predict_scores(dataloader, model):
    predictions = []
    device = torch.device('cuda') 
   
    model.to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids, attention_mask, targets = (item.to(device) for item in batch)
            outputs = model(input_ids, attention_mask=attention_mask)
            batch_logits = outputs.logits
            
            # Convert tensor to list
            if batch_logits.shape[0] == 1:
                pred = [float(batch_logits.cpu().numpy())]  
            else:
                pred = batch_logits.squeeze().cpu().tolist() 

            predictions.extend(pred)
            
    return predictions


def get_and_save_top_k_sentences_with_token_length(val_df, k_values, tokenizer,result_path,dtype):
    for k in k_values:
        results = []
        for name, group in val_df.groupby('ReviewID'):
            sorted_group = group.sort_values(by='predicted_score', ascending=False).head(k)
            concatenated_sent_id = list(sorted_group['sent_id'])
            concatenated_sentences = " ".join(sorted_group['docs_sent'])
            concatenated_predicted_score = list(sorted_group['predicted_score'])
            first_target = group['ground_truth_summary'].iloc[0]
            
            # Calculate BART_token_len for this concatenated_sentences
            bart_token_len = len(tokenizer.encode(concatenated_sentences))
            
            results.append({
                'ReviewID': name,
                'sent_id': concatenated_sent_id,
                'top_k_sentences': concatenated_sentences,
                'predicted_score': concatenated_predicted_score,
                'ground_truth_summary': first_target,
                'BART_token_len': bart_token_len  # Adding BART_token_len here
            })

        top_k_df = pd.DataFrame(results)
        excel_name = os.path.join(result_path, f"Top_{k}_sentences_with_token_length_for_each_ReviewID_{dtype}.csv")
        top_k_df.to_csv(excel_name, index=False)

def plot_and_save_boxplot(dfs, save_path='boxplot_token_lengths.png'):
    """
    Plots and saves a boxplot showing the distribution of token lengths for different top_k values.
    
    Parameters:
    - dfs: list of pd.DataFrame
        List of dataframes containing the token length data.
    - save_path: str
        The path where the boxplot will be saved.

    """
    # Combine dataframes
    dfs[0]['top_k'] = 2
   
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df['BART_token_len'] = pd.to_numeric(combined_df['BART_token_len'], errors='coerce')

    plt.figure(figsize=(10, 6))
    plt.plot([0, 2], [1024, 1024], color='red', label='Max input tokens of BART model (1024)')
    plt.legend()

    sns.boxplot(x='top_k', y='BART_token_len', data=combined_df)

    plt.title('Distribution of Token Lengths for Different top_k Values')
    plt.xlabel('top_k Value')
    plt.ylabel('Token Length')

    plt.savefig(save_path)

    plt.show()


def calculate_rouge_for_each_review(df):
    # Initialize Rouge
    rouge = Rouge()
    
    # Get all unique ReviewIDs
    unique_review_ids = df['ReviewID'].unique()
    rouge_scores = {}

    for review_id in unique_review_ids:
        filtered_df = df[df['ReviewID'] == review_id]
        docs_sent_aggregated = " ".join(filtered_df['docs_sent'])
        target_aggregated = filtered_df['ground_truth_summary'].iloc[0]
        scores = rouge.get_scores(docs_sent_aggregated, target_aggregated)[0]
        rouge_scores[review_id] = scores

    return rouge_scores

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

from rouge_score import rouge_scorer



def cal_rouge(df):
    # Extract 'Target' and 'top_k_sentences' columns from the DataFrame
    refs = df['ground_truth_summary'].tolist()
    hyps = df['top_k_sentences'].tolist()

    # Initialize ROUGE scorer
    metrics = ['rouge1', 'rouge2', 'rougeL']
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)

    if len(refs) != len(hyps):
        raise Exception(f"No. of Refs and Hyps are not equal")

    # Initialize results dictionary to hold average scores
    results = {}
    for metric in metrics:
        results[metric] = {'f': [], 'p': [], 'r': []}
        
    # Calculate scores for each reference and hypothesis pair
    for idx in range(len(refs)):
        scores = scorer.score(refs[idx].strip(), hyps[idx].strip())
        for metric in metrics:
            results[metric]['f'].append(scores[metric].fmeasure)
            results[metric]['p'].append(scores[metric].precision)
            results[metric]['r'].append(scores[metric].recall)

    # Calculate the average for each metric and each type (f, p, r)
    avg_results = {}
    for metric in metrics:
        avg_results[metric] = {
            'f': np.average(results[metric]['f']),
            'p': np.average(results[metric]['p']),
            'r': np.average(results[metric]['r'])
        }

    return avg_results