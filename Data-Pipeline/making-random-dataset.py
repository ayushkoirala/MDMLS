
import os
import pandas as pd
import random
from transformers import AutoTokenizer, AutoModelForTokenClassification
import pandas as pd
import os
import re
import torch
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
def get_rouge(candidate, reference):
    scores = {}  # Initialize scores as an empty dictionary
    rouge = Rouge()
    try:
        scores = rouge.get_scores(candidate, reference, avg='avg')
        scores = {f"R{k1[-1]}_{k2}": v2 for k1, v1 in scores.items() for k2, v2 in v1.items() if k2 != 'f'}
    except Exception as e:
        print('Exception:', e)
        scores = {f"R1_{metric}": None for metric in ['r', 'p']}
        scores.update({f"R2_{metric}": None for metric in ['r', 'p']})
        scores.update({f"Rl_{metric}": None for metric in ['r', 'p']})
        
    return scores
    
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
    
def get_rouge_scores(df):
    r1_r = []
    r1_p = []
    r2_r = []
    r2_p = []
    rl_r = []
    rl_p = []
    df = df[(df['docs_sent'] != '.') & (df['docs_sent'] != '')]
    for index, row in df.iterrows():
        candidate = row['docs_sent']
        target = row['ground_truth_summary']
        scores = get_rouge(candidate, target)
    
        r1_r.append(scores.get('R1_r', None))
        r1_p.append(scores.get('R1_p', None))
        r2_r.append(scores.get('R2_r', None))
        r2_p.append(scores.get('R2_p', None))
        rl_r.append(scores.get('Rl_r', None))
        rl_p.append(scores.get('Rl_p', None))
    
    df['R1_r'] = r1_r
    df['R1_p'] = r1_p
    df['R2_r'] = r2_r
    df['R2_p'] = r2_p
    df['Rl_r'] = rl_r
    df['Rl_p'] = rl_p
    return df

def calculate_final_score(row, lamb=0.15):
    f1_score = {}
    for r in ['R1', 'R2']:
        div = row[f'{r}_p'] + row[f'{r}_r']
        f1_score[r] = 0 if div == 0 else 2 * (row[f'{r}_p'] * row[f'{r}_r']) / div
    return (lamb * f1_score['R1']) + f1_score['R2']

def preprocess_dataframe(df):
    result_df1 = []
    for i, row in df.iterrows():
        sentences = sent_tokenize(row['Candidate document'])
        for j, sent in enumerate(sentences):
            if sent:
                sent_id = f"{row['ReviewID']}-{row['PMID']}-{str(j).zfill(2)}"
                input_value = f"[CLS]{row['Masked Background']}[SEP]{sent}[SEP]"
                new_row = {
                    'ReviewID': row['ReviewID'],
                    'PMID': row['PMID'],
                    'sent_id': sent_id,
                    'docs_sent': sent,
                    'Masked Background': row['Masked Background'],
                    'Background':row['Background'],
                    'ground_truth_summary': row['ground_truth_summary'],
                    'input_seq': input_value  
                }
                result_df1.append(new_row)
    return pd.DataFrame(result_df1).reset_index(drop=True)
# def randomly_mask_text(text, ratio, avg_masked_tokens,priority_words=None):
#     if priority_words is None:
#         priority_words = ["of", "and", "in", "with", "on", "for", "but", "or", "so", "because", "although", "if", "at", "by", "from", "to"]
    
#     words = text.split()
#     num_words_to_mask = max(1, int(len(words) * ratio))

#     # Identify indices of priority words
#     priority_indices = [i for i, word in enumerate(words) if word.lower() in priority_words]

#     # If there are more priority words than the number of words to mask, then just mask the priority words
#     if len(priority_indices) >= num_words_to_mask:
#         masked_indices = random.sample(priority_indices, num_words_to_mask)
#     else:
#         # First mask all priority words
#         masked_indices = priority_indices
#         remaining_to_mask = num_words_to_mask - len(priority_indices)
        
#         # Mask the remaining random words
#         non_priority_indices = list(set(range(len(words))) - set(priority_indices))
#         masked_indices += random.sample(non_priority_indices, remaining_to_mask)

#     for idx in masked_indices:
#         words[idx] = '[MASK]'

#     return ' '.join(words)

def randomly_mask_text(text, ratio, avg_masked_tokens, priority_words=None):
    if priority_words is None:
        priority_words = ["of", "and", "in", "with", "on", "for", "but", "or", "so", "because", "although", "if", "at", "by", "from", "to"]
    
    words = text.split()
    tokens_masked = 0
    num_words_to_mask = max(1, int(len(words) * ratio))

    # Identify indices of priority words
    priority_indices = [i for i, word in enumerate(words) if word.lower() in priority_words]

    # Mask the priority words first based on the average masked tokens
    for idx in priority_indices:
        if tokens_masked < avg_masked_tokens:
            words[idx] = '[MASK]'
            tokens_masked += 1

    # If we have already masked the average number of tokens or more, return
    if tokens_masked >= avg_masked_tokens:
        return ' '.join(words)

    # Calculate how many more tokens we can mask
    remaining_masks = avg_masked_tokens - tokens_masked

    # If there are more priority words than the number of words to mask, then just mask the priority words
    if len(priority_indices) >= num_words_to_mask:
        masked_indices = random.sample(priority_indices, remaining_masks)
    else:
        # Mask the remaining random words
        non_priority_indices = list(set(range(len(words))) - set(priority_indices))
        masked_indices = random.sample(non_priority_indices, remaining_masks)

    for idx in masked_indices:
        words[idx] = '[MASK]'

    return ' '.join(words)



def preprocess_and_save(input_filepath, output_dir, mask_ratio,avg_masked_tokens):
    # Load dataframe
    df = pd.read_csv(input_filepath)
    
    # Apply masking and store in new column
    #df['Masked Background'] = df['Background'].apply(preprocess_background, args=(tokenizer, model, device, mask_ratio))
    
    df['Masked Background'] = df['Background'].apply(randomly_mask_text, args=(mask_ratio,avg_masked_tokens,))
    display("Masking background using PICO Done")
    
    
    # Save the dataframe
    df_preprocess = preprocess_dataframe(df)
    display("spliting into sentences")
    rouge_scores_df = get_rouge_scores(df_preprocess)
    display("Getting Target R1,R2 and RL")
    rouge_scores_df['final_score'] = rouge_scores_df.apply(calculate_final_score, axis=1)
    display("Final Score is saved")
    rouge_scores_df.drop(['R1_r', 'R2_r', 'R1_p', 'R2_p', 'Rl_p', 'Rl_r'], axis=1, inplace=True)
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filename = os.path.join(output_dir, os.path.basename(input_filepath))
    rouge_scores_df.to_csv(output_filename, index=False)


def main():
    parser = argparse.ArgumentParser(description="Data Preprocesing")
    parser.add_argument('--log', action='store_true',help='logging')
    parser.add_argument('--log_dir', type=str, default="preprocess", help='Logging directory')
    parser.add_argument('--full', action='store_true', help='Run on full data')
    #data_path = '/home/ayushk/work/Data-preprocessing-pipeline/Dataset'
    parser.add_argument('--input_dir', type=str, default='/home/akoirala/Thesis/Data-making-pipeline/dataset', help='Path to the dataset')
    parser.add_argument('--result', type=str, default='/home/akoirala/Thesis/Data-Pipeline/random_dataset', help='Path to the result dataset')
    
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
    display(f"{args=}")
    
    files = ["top_6_dev.csv", "top_6_test.csv","top_6_train.csv"]
    ratios = [0.15, 0.3, 0.45]

    for ratio in ratios:
        output_directory = os.path.join(args.result,f"random_{ratio}")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        for filename in files:
            if filename == 'top_6_dev.csv':
                average_token_count =  12.8053
                avg_masked_tokens = round(average_token_count * ratio)
                input_filepath = os.path.join(args.input_dir, filename)
                preprocess_and_save(input_filepath, output_directory, ratio,avg_masked_tokens)
            elif filename == 'top_6_test.csv':
                average_token_count =  12.7125
                avg_masked_tokens = round(average_token_count * ratio)
                input_filepath = os.path.join(args.input_dir, filename)
                preprocess_and_save(input_filepath, output_directory, ratio,avg_masked_tokens)
            elif filename == 'top_6_train.csv':
                average_token_count =  13.4421
                avg_masked_tokens = round(average_token_count * ratio)
                input_filepath = os.path.join(args.input_dir, filename)
                preprocess_and_save(input_filepath, output_directory, ratio,avg_masked_tokens)
            
            
if __name__ == "__main__":
    main()
