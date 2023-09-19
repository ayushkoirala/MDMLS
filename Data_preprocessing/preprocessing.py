import pandas as pd
import os
import re
import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
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


def filter_and_concat_reviews(train_input_path, dev_target_path, max_abstract_count=30):
    train_input_df = pd.read_csv(train_input_path)
    filtered_train_input_df = train_input_df.groupby('ReviewID').filter(lambda x: len(x) <= max_abstract_count)

    dev_target_df = pd.read_csv(dev_target_path)
    filtered_review_ids = filtered_train_input_df['ReviewID'].unique()
    filtered_dev_target_df = dev_target_df[dev_target_df['ReviewID'].isin(filtered_review_ids)]

    merged_df = pd.merge(filtered_train_input_df, filtered_dev_target_df, on='ReviewID', how='inner')
    merged_df.dropna(subset=['Background', 'Target'], inplace=True)
    return merged_df

def remove_hash_concat(row):
    def remove_unwanted_chars(text):
        if isinstance(text, str):
            cleaned_text = text.replace('#', '').replace('\n', '')
            return cleaned_text
        return text

    for col in ['Background', 'Target', 'Abstract']:
        row[col] = remove_unwanted_chars(row[col])
    return row

# Data Retrieval Function
def top_k_document(sample_train_inputs,top_k,device):
    context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

    
    context_encoder.to(device)

    

    timings = {}
    desired_output = []
    batch_size = 5

    for review_id in sample_train_inputs['ReviewID'].unique():
        start_time = time.time()
        review_entries = sample_train_inputs[sample_train_inputs['ReviewID'] == review_id]
        review_abstracts = review_entries['Abstract'].tolist()
        background_data = review_entries['Background'].tolist()[0]

        retrieval_scores_list = []
        batch_abstracts = []

        for abstract in review_abstracts:
            batch_abstracts.append(abstract)
            if len(batch_abstracts) == batch_size:
                abstract_tokens = context_tokenizer(batch_abstracts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                multi_document_embeddings = context_encoder(**abstract_tokens).pooler_output

                background_tokens = context_tokenizer([background_data], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                background_embeddings = context_encoder(**background_tokens).pooler_output

                retrieval_scores = torch.mm(multi_document_embeddings, background_embeddings.T)
                retrieval_scores_list.append(retrieval_scores)

                batch_abstracts = []

        if batch_abstracts:
            abstract_tokens = context_tokenizer(batch_abstracts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            multi_document_embeddings = context_encoder(**abstract_tokens).pooler_output
            retrieval_scores = torch.mm(multi_document_embeddings, background_embeddings.T)
            retrieval_scores_list.append(retrieval_scores)

        retrieval_scores = torch.cat(retrieval_scores_list)
        end_time = time.time()
        elapsed_time = end_time - start_time
        timings[review_id] = elapsed_time

        k = min(top_k, retrieval_scores.shape[0])
        top_k_indices = torch.argsort(retrieval_scores.view(-1), descending=True)[:k]

        for idx in top_k_indices:
            idx = idx.item()
            top_k_candidate_pmids = review_entries.iloc[idx]['PMID']
            top_k_candidate_document = review_entries.iloc[idx]['Abstract']
            top_k_candidate_background = background_data
            top_k_candidate_target = review_entries.iloc[idx]['Target']

            desired_output.append({
                'ReviewID': review_id,
                'PMID': top_k_candidate_pmids,
                'Background': top_k_candidate_background,
                'Candidate document': top_k_candidate_document,
                'ground_truth_summary':top_k_candidate_target
            })

    desired_output_df = pd.DataFrame(desired_output)
    return desired_output_df


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


def preprocess_background(background, tokenizer, model, device):
    background = background.replace("(", "").replace(")", "").replace(",", "").replace("\n", " ")
    inputs = tokenizer.encode(background, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    outputs = model(inputs).logits
    predictions = outputs.argmax(dim=2)[0]

    tokens = tokenizer.convert_ids_to_tokens(inputs[0])
    masked_tokens = []
    for token, prediction in zip(tokens, predictions):
        label = model.config.id2label[prediction.item()]

        if label != 'O' and random.random() < 0.45:
            token = tokenizer.mask_token
            
        if token not in ['[CLS]', '[SEP]']:
            masked_tokens.append(token)

    return " ".join(masked_tokens)

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
                    'ground_truth_summary': row['ground_truth_summary'],
                    'input_seq': input_value  
                }
                result_df1.append(new_row)
    return pd.DataFrame(result_df1).reset_index(drop=True)




def calculate_final_score(row, lamb=0.15):
    f1_score = {}
    for r in ['R1', 'R2']:
        div = row[f'{r}_p'] + row[f'{r}_r']
        f1_score[r] = 0 if div == 0 else 2 * (row[f'{r}_p'] * row[f'{r}_r']) / div
    return (lamb * f1_score['R1']) + f1_score['R2']


def process_data(data_type, k, device,args):
    display("Process_data function starts")
    input_df = filter_and_concat_reviews(
        train_input_path=os.path.join(args.data_path, f'{data_type}-inputs.csv'),
        dev_target_path=os.path.join(args.data_path, f'{data_type}-targets.csv')
    )
    display("input_df dataframe is done")
    input_df = input_df.apply(remove_hash_concat, axis=1)
    input_df.to_csv(os.path.join(args.result, f'{data_type}-inputs-30.csv'), index=False)
    display("Top 30 document for each ReviewId is saved")

    top_k_document_df = top_k_document(input_df, k, device)
    top_k_document_df.to_csv(os.path.join(args.result, f'top_k_{data_type}_document.csv'), index=False)
    display("Top K (k=6) document for each ReviewId is saved")

    tokenizer = AutoTokenizer.from_pretrained("kamalkraj/BioELECTRA-PICO")
    model = AutoModelForTokenClassification.from_pretrained("kamalkraj/BioELECTRA-PICO").to(device)
    
    top_k_document_df['Masked Background'] = top_k_document_df['Background'].apply(preprocess_background, args=(tokenizer, model, device))
    display("Masking background using PICO Done")
    
    df_exploded = preprocess_dataframe(top_k_document_df)
    rouge_scores_df = get_rouge_scores(df_exploded)
    display("Getting Target R1,R2 and RL")
    
    rouge_scores_df['final_score'] = rouge_scores_df.apply(calculate_final_score, axis=1)
    display("Final Score is saved")

    rouge_scores_df.drop(['R1_r', 'R2_r', 'R1_p', 'R2_p', 'Rl_p', 'Rl_r'], axis=1, inplace=True)
    
    rouge_scores_df.to_csv(os.path.join(args.result, f'PICO_0.15_{data_type}.csv'), index=False)
    display("Process_data function Done")
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

def main():
    parser = argparse.ArgumentParser(description="Data Preprocesing")
    parser.add_argument('--log', action='store_true',help='logging')
    parser.add_argument('--log_dir', type=str, default="preprocess", help='Logging directory')
    parser.add_argument('--full', action='store_true', help='Run on full data')
    #data_path = '/home/ayushk/work/Data-preprocessing-pipeline/Dataset'
    parser.add_argument('--data_path', type=str, default='/home/akoirala/Thesis/Dataset', help='Path to the dataset')
    parser.add_argument('--result', type=str, default='/home/akoirala/Thesis/Data-Preprocessing-Pipeline/result', help='Path to the result dataset')
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

    process_main = log_process(f"Program Running", log_=args.log, print_=True)
    process = log_process(f"Loading data", log_=args.log, print_=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    k = 6
    
    #process_data('train', k, device,args)
    process_data('dev', k, device,args)
    process.end()
    process_main.end()
    

if __name__ == "__main__":
    main()
