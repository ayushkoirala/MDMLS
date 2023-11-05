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


def filter_and_concat_reviews(train_input_path, dev_target_path):
    train_input_df = pd.read_csv(train_input_path)
    # filtered_train_input_df = train_input_df.groupby('ReviewID').filter(lambda x: len(x) <= max_abstract_count)

    dev_target_df = pd.read_csv(dev_target_path)
    filtered_review_ids = train_input_df['ReviewID'].unique()
    filtered_dev_target_df = dev_target_df[dev_target_df['ReviewID'].isin(filtered_review_ids)]

    merged_df = pd.merge(train_input_df, filtered_dev_target_df, on='ReviewID', how='inner')
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

import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import os
import time
import pandas as pd

def top_k_document(sample_train_inputs, device, args, top_k=6):
    context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

    background_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    background_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    batch_size = min(args.top_k_batch_size, torch.cuda.get_device_properties(device).total_memory / (512 ** 2))
    print(batch_size)
    context_encoder = context_encoder.half().to(device)
    background_encoder = background_encoder.half().to(device)

    timings = {}
    desired_output = []

    for review_id in sample_train_inputs['ReviewID'].unique():
        start_time = time.time()
        review_entries = sample_train_inputs[sample_train_inputs['ReviewID'] == review_id]
        review_abstracts = review_entries['Abstract'].tolist()
        background_data = review_entries['Target'].tolist()[0]

        retrieval_scores_list = []
        batch_abstracts = []

        for abstract in review_abstracts:
            batch_abstracts.append(abstract)
            if len(batch_abstracts) == batch_size or abstract is review_abstracts[-1]:
                abstract_tokens = context_tokenizer(batch_abstracts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                background_tokens = background_tokenizer([background_data], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                multi_document_embeddings = context_encoder(**abstract_tokens).pooler_output
                background_embeddings = background_encoder(**background_tokens).pooler_output
                retrieval_scores = torch.mm(multi_document_embeddings, background_embeddings.T)
                retrieval_scores_list.append(retrieval_scores)
                del abstract_tokens, multi_document_embeddings, background_tokens, background_embeddings, retrieval_scores
                torch.cuda.empty_cache()
                batch_abstracts = []

        retrieval_scores = torch.cat(retrieval_scores_list)
        end_time = time.time()
        timings[review_id] = end_time - start_time
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
                'ground_truth_summary': top_k_candidate_target
            })
    desired_output_df = pd.DataFrame(desired_output)
    return desired_output_df


import argparse
import os
import torch
import pandas as pd
import logging
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Data Preprocessing")
    parser.add_argument('--log', action='store_true', help='logging')
    parser.add_argument('--log_dir', type=str, default="preprocess", help='Logging directory')
    parser.add_argument('--full', action='store_true', help='Run on full data')
    parser.add_argument('--data_path', type=str, default='/home/akoirala/Thesis/Data-making-pipeline/main_dataset', help='Path to the dataset')
    parser.add_argument('--create_dataset', type=str, default='/home/akoirala/Thesis/Data-making-pipeline/main_dataset/result', help='Path to the result dataset')
    parser.add_argument('--top_k_batch_size', type=int, default=1, help='DPR batch size')
    parser.add_argument('--mask_ratio', type=list, default=[0.15,0.3,0.45], help='Mask ratio for data augmentation')
    args = parser.parse_args()

    log_dir = args.log_dir + "_prototype" if not args.full else args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = "native_" + datetime.now().strftime("%Y_%m_%d-%H_%M")
    logging.basicConfig(
        filename=f'{log_dir}/{log_filename}.log',
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info(f"{args=}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    dataset_type = 'dev'
    dataset_file_path = os.path.join(args.data_path, f'{dataset_type}-inputs-full.csv')
    dataset_df = pd.read_csv(dataset_file_path)

    logging.info("Calculating top k documents...")
    top_k_document_df = top_k_document(dataset_df, device=device, args=args)
    
    output_file_path = os.path.join(args.create_dataset, f'top_k_{dataset_type}_document.csv')
    top_k_document_df.to_csv(output_file_path, index=False)
    logging.info(f"Saved top_k_document for {dataset_type} to {output_file_path}")
    

if __name__ == "__main__":
    main()
