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


def create_test(input_df,test_path):
    selected_reviewIDs = input_df['ReviewID'].unique()
    selected_reviewIDs = pd.Series(selected_reviewIDs).sample(n=400, random_state=1)

    # Filter the DataFrame to create the test set with the selected ReviewID's
    test_df = input_df[input_df['ReviewID'].isin(selected_reviewIDs)]
    test_reviewIDs = test_df['ReviewID'].unique()
    common_in_train = input_df['ReviewID'].isin(test_reviewIDs)
    input_df = input_df[~common_in_train]
    # Save the test set to a new CSV file
    test_df.to_csv(test_path, index=False)
    return test_df,input_df

def creating_dataset(data_type, device,args):
    display("Process_data function starts")
    input_df = filter_and_concat_reviews(
        train_input_path=os.path.join(args.data_path, f'{data_type}-inputs.csv'),
        dev_target_path=os.path.join(args.data_path, f'{data_type}-targets.csv')
    )
    display("input_df dataframe is done")
    input_df = input_df.apply(remove_hash_concat, axis=1)
    input_df.to_csv(os.path.join(args.create_dataset, f'{data_type}-inputs-full.csv'), index=False)
    display("Top candidate document for each ReviewId is saved")
    if data_type == 'train':
        test_df,input_df = create_test(input_df,os.path.join(args.create_dataset,'test-inputs-full.csv'))
        return test_df, input_df
    return input_df


# def top_k_document(sample_train_inputs, device,args,top_k=6):

#     context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
#     context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

#     background_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
#     background_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

#     # Move models to the specified device
#     gpu_ids = [int(id) for id in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if id.strip()]
#     if len(gpu_ids) > 1:
#         print(f"Let's use {len(gpu_ids)} GPUs!")
#         context_encoder = context_encoder.half().to(device)
#         background_encoder = background_encoder.half().to(device)

#         context_encoder = torch.nn.DataParallel(context_encoder, device_ids=gpu_ids)
#         background_encoder = torch.nn.DataParallel(background_encoder, device_ids=gpu_ids)


#     timings = {}
#     desired_output = []

#     batch_size = args.top_k_batch_size

#     for review_id in sample_train_inputs['ReviewID'].unique():
#         start_time = time.time()
#         review_entries = sample_train_inputs[sample_train_inputs['ReviewID'] == review_id]
#         review_abstracts = review_entries['Abstract'].tolist()
#         background_data = review_entries['Target'].tolist()[0]

#         retrieval_scores_list = []
#         batch_abstracts = []

#         for abstract in review_abstracts:
#             batch_abstracts.append(abstract)
#             if len(batch_abstracts) == batch_size or abstract is review_abstracts[-1]:
#                 # Calculate max_length for abstract_tokens
#                 max_length_abstract = min(512, len(max(batch_abstracts, key=len)))
#                 abstract_tokens = context_tokenizer(batch_abstracts, return_tensors="pt", padding=True, truncation=True, max_length=max_length_abstract).to(device)
#                 multi_document_embeddings = context_encoder(**abstract_tokens).pooler_output
                
#                 # Calculate max_length for background_tokens
#                 max_length_background = min(512, len(background_data))
#                 background_tokens = background_tokenizer([background_data], return_tensors="pt", padding=True, truncation=True, max_length=max_length_background).to(device)
#                 background_embeddings = background_encoder(**background_tokens).pooler_output

#                 retrieval_scores = torch.mm(multi_document_embeddings, background_embeddings.T)
#                 retrieval_scores_list.append(retrieval_scores)

#                 batch_abstracts = []

#         # If there are any remaining abstracts in batch_abstracts, process them as well
#         if batch_abstracts:
#             max_length_abstract = min(512, len(max(batch_abstracts, key=len)))
#             abstract_tokens = context_tokenizer(batch_abstracts, return_tensors="pt", padding=True, truncation=True, max_length=max_length_abstract).to(device)
#             multi_document_embeddings = context_encoder(**abstract_tokens).pooler_output
#             retrieval_scores = torch.mm(multi_document_embeddings, background_embeddings.T)
#             retrieval_scores_list.append(retrieval_scores)

#         retrieval_scores = torch.cat(retrieval_scores_list)
#         end_time = time.time()
#         elapsed_time = end_time - start_time
#         timings[review_id] = elapsed_time

#         k = min(top_k, retrieval_scores.shape[0])
#         top_k_indices = torch.argsort(retrieval_scores.view(-1), descending=True)[:k]

#         for idx in top_k_indices:
#             idx = idx.item()
#             top_k_candidate_pmids = review_entries.iloc[idx]['PMID']
#             top_k_candidate_document = review_entries.iloc[idx]['Abstract']
#             top_k_candidate_background = background_data
#             top_k_candidate_target = review_entries.iloc[idx]['Target']

#             desired_output.append({
#                 'ReviewID': review_id,
#                 'PMID': top_k_candidate_pmids,
#                 'Background': top_k_candidate_background,
#                 'Candidate document': top_k_candidate_document,
#                 'ground_truth_summary': top_k_candidate_target
#             })

#     desired_output_df = pd.DataFrame(desired_output)
#     return desired_output_df


from tqdm import tqdm
from IPython.display import display
import pandas as pd
import torch
import time
import os
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from torch.utils.data import DataLoader, TensorDataset

# def top_k_document(sample_train_inputs, device, args, top_k=6):
#     context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
#     context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    
#     background_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
#     background_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

#     gpu_ids = [int(id) for id in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if id.strip()]
#     if len(gpu_ids) > 1:
#         print(f"Let's use {len(gpu_ids)} GPUs!")
#         context_encoder = context_encoder.half().to(device)
#         background_encoder = background_encoder.half().to(device)

#         context_encoder = torch.nn.DataParallel(context_encoder, device_ids=gpu_ids)
#         background_encoder = torch.nn.DataParallel(background_encoder, device_ids=gpu_ids)

#     timings = {}
#     desired_output = []
#     batch_size = args.top_k_batch_size

#     review_ids = sample_train_inputs['ReviewID'].unique()

#     for review_id in tqdm(review_ids, desc="Processing Reviews"):
#         torch.cuda.empty_cache()  
#         start_time = time.time()
        
#         review_entries = sample_train_inputs[sample_train_inputs['ReviewID'] == review_id]
#         review_abstracts = review_entries['Abstract'].tolist()
#         background_data = review_entries['Target'].tolist()[0]

#         retrieval_scores_list = []
#         batch_abstracts = []
#         for abstract in tqdm(review_abstracts, desc=f"Processing Abstracts for Review {review_id}", leave=False):
#             batch_abstracts.append(abstract)
#             if len(batch_abstracts) == batch_size or abstract is review_abstracts[-1]:
#                 torch.cuda.empty_cache()
#                 # Calculate max_length for abstract_tokens
#                 max_length_abstract = min(512, len(max(batch_abstracts, key=len)))
#                 abstract_tokens = context_tokenizer(batch_abstracts, return_tensors="pt", padding=True, truncation=True, max_length=max_length_abstract).to(device)
#                 multi_document_embeddings = context_encoder(**abstract_tokens).pooler_output
                
#                 # Calculate max_length for background_tokens
#                 max_length_background = min(512, len(background_data))
#                 background_tokens = background_tokenizer([background_data], return_tensors="pt", padding=True, truncation=True, max_length=max_length_background).to(device)
#                 background_embeddings = background_encoder(**background_tokens).pooler_output
#                 retrieval_scores = torch.mm(multi_document_embeddings, background_embeddings.T)
#                 retrieval_scores_list.append(retrieval_scores)

#                 batch_abstracts = []

#         if batch_abstracts:
#             max_length_abstract = min(512, len(max(batch_abstracts, key=len)))
#             abstract_tokens = context_tokenizer(batch_abstracts, return_tensors="pt", padding=True, truncation=True, max_length=max_length_abstract).to(device)
#             multi_document_embeddings = context_encoder(**abstract_tokens).pooler_output
#             retrieval_scores = torch.mm(multi_document_embeddings, background_embeddings.T)
#             retrieval_scores_list.append(retrieval_scores)

#         retrieval_scores = torch.cat(retrieval_scores_list)
#         end_time = time.time()
#         elapsed_time = end_time - start_time
#         timings[review_id] = elapsed_time

#         k = min(top_k, retrieval_scores.shape[0])
#         top_k_indices = torch.argsort(retrieval_scores.view(-1), descending=True)[:k]

#         for idx in top_k_indices:
#             idx = idx.item()
#             top_k_candidate_pmids = review_entries.iloc[idx]['PMID']
#             top_k_candidate_document = review_entries.iloc[idx]['Abstract']
#             top_k_candidate_background = background_data
#             top_k_candidate_target = review_entries.iloc[idx]['Target']

#             desired_output.append({
#                 'ReviewID': review_id,
#                 'PMID': top_k_candidate_pmids,
#                 'Background': top_k_candidate_background,
#                 'Candidate document': top_k_candidate_document,
#                 'ground_truth_summary': top_k_candidate_target
#             })


#         display(f"Completed Review {review_id}")  # Display completion message for each review

#     desired_output_df = pd.DataFrame(desired_output)
#     return desired_output_df

# def top_k_document(sample_train_inputs, device, args, top_k=3):
#     context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device)
#     context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
#     background_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)
#     background_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    
#     # If multiple GPUs are available, use DataParallel
#     if torch.cuda.device_count() > 1:
#         context_encoder = torch.nn.DataParallel(context_encoder)
#         background_encoder = torch.nn.DataParallel(background_encoder)

#     desired_output = []
#     batch_size = args.top_k_batch_size
#     review_ids = sample_train_inputs['ReviewID'].unique()
    
#     for review_id in tqdm(review_ids, desc="Processing Reviews"):
#         print(review_id)
#         torch.cuda.empty_cache()  
#         start_time = time.time()
        
#         review_entries = sample_train_inputs[sample_train_inputs['ReviewID'] == review_id]
#         review_abstracts = review_entries['Abstract'].tolist()
#         background_data = review_entries['Target'].tolist()[0]
        
#         # Creating a DataLoader
#         dataset = TensorDataset(torch.tensor(range(len(review_abstracts))))
#         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
#         retrieval_scores_list = []
#         for batch in tqdm(dataloader, desc=f"Processing Abstracts for Review {review_id}", leave=False):
#             batch_abstracts = [review_abstracts[idx] for idx in batch[0]]
#             torch.cuda.empty_cache()
#             max_length_abstract = min(512, len(max(batch_abstracts, key=len)))
#             abstract_tokens = context_tokenizer(batch_abstracts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
#             multi_document_embeddings = context_encoder(**abstract_tokens).pooler_output
#             max_length_background = min(512, len(background_data))
#             background_tokens = background_tokenizer([background_data], return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
#             background_embeddings = background_encoder(**background_tokens).pooler_output
#             retrieval_scores = torch.mm(multi_document_embeddings, background_embeddings.T)
#             retrieval_scores_list.append(retrieval_scores)

#         retrieval_scores = torch.cat(retrieval_scores_list)
#         end_time = time.time()
#         elapsed_time = end_time - start_time

#         k = min(top_k, retrieval_scores.shape[0])
#         top_k_indices = torch.argsort(retrieval_scores.view(-1), descending=True)[:k]

#         for idx in top_k_indices:
#             idx = idx.item()
#             top_k_candidate_pmids = review_entries.iloc[idx]['PMID']
#             top_k_candidate_document = review_entries.iloc[idx]['Abstract']
#             top_k_candidate_background = background_data
#             top_k_candidate_target = review_entries.iloc[idx]['Target']
#             desired_output.append({
#                 'ReviewID': review_id,
#                 'PMID': top_k_candidate_pmids,
#                 'Background': top_k_candidate_background,
#                 'Candidate document': top_k_candidate_document,
#                 'ground_truth_summary': top_k_candidate_target
#             })
    
#     desired_output_df = pd.DataFrame(desired_output)
#     return desired_output_df

import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from transformers import (DPRContextEncoder, DPRContextEncoderTokenizer, 
                          DPRQuestionEncoder, DPRQuestionEncoderTokenizer)
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

def top_k_document(gpu, ngpus_per_node, sample_train_inputs, device, args, top_k=6):
    torch.cuda.empty_cache()  
    # Initialize distributed environment
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    # os.environ['RANK'] = '0'
    # os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
   
    
    dist.init_process_group(backend='nccl',init_method='tcp://localhost:23456', rank=gpu, world_size=ngpus_per_node)

    
    context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device)
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    background_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)
    background_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

    
    context_encoder = context_encoder.half().to(device)
    background_encoder = background_encoder.half().to(device)
    context_encoder = DistributedDataParallel(context_encoder,device_ids=[gpu])
    background_encoder = DistributedDataParallel(background_encoder,device_ids=[gpu])
    desired_output = []
    batch_size = 1
    review_ids = sample_train_inputs['ReviewID'].unique()
    for review_id in tqdm(review_ids, desc="Processing Reviews"):
        torch.cuda.empty_cache()  
        start_time = time.time()
        review_entries = sample_train_inputs[sample_train_inputs['ReviewID'] == review_id]
        review_abstracts = review_entries['Abstract'].tolist()
        background_data = review_entries['Target'].tolist()[0]
        dataset = TensorDataset(torch.tensor(range(len(review_abstracts))))
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
        retrieval_scores_list = []
        for batch in tqdm(dataloader, desc=f"Processing Abstracts for Review {review_id}", leave=False):
            torch.cuda.empty_cache()
            batch_abstracts = [review_abstracts[idx] for idx in batch[0]]
            
            max_length_abstract = min(512, len(max(batch_abstracts, key=len)))
            abstract_tokens = context_tokenizer(batch_abstracts, return_tensors="pt", padding=True, truncation=True, max_length=max_length_abstract).to(device)
            multi_document_embeddings = context_encoder(**abstract_tokens).pooler_output
            max_length_background = min(512, len(background_data))
            background_tokens = background_tokenizer([background_data], return_tensors="pt", padding=True, truncation=True, max_length=max_length_background).to(device)
            background_embeddings = background_encoder(**background_tokens).pooler_output
            retrieval_scores = torch.mm(multi_document_embeddings, background_embeddings.T)
            retrieval_scores_list.append(retrieval_scores)
        retrieval_scores = torch.cat(retrieval_scores_list)
        end_time = time.time()
        elapsed_time = end_time - start_time
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
    output_file_path = os.path.join(args.create_dataset,f'top_k_val11_document.csv')
    desired_output_df.to_csv(output_file_path, index=False)



import torch.multiprocessing as mp
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
    display(f"{args=}")

    process_main = log_process(f"Program Running", log_=args.log, print_=True)
    process = log_process(f"Loading data", log_=args.log, print_=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_type= 'dev'
    dataset_df = pd.read_csv(r'/home/akoirala/Thesis/Data-making-pipeline/main_dataset/dev-inputs-full.csv')
    ngpus_per_node = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mp.spawn(top_k_document, nprocs=ngpus_per_node, args=(ngpus_per_node, dataset_df, device, args))
    # torch.cuda.empty_cache()  
    # top_k_document_df = top_k_document(dataset_df, device=device, args=args)
    # output_file_path = os.path.join(args.create_dataset,f'top_k_{dataset_type}_document.csv')
    # top_k_document_df.to_csv(output_file_path, index=False)
    # display(f"Saved top_k_document for {dataset_type} to {output_file_path}")
    # dataset_types=['train']
    # for data_type in dataset_types:
    #     dataset_df = creating_dataset(data_type, device=device, args=args)    
    #     display(f"{data_type} dataframe created")
    #     top_k_document_df = top_k_document(dataset_df, device=device, args=args)
    #     output_file_path = os.path.join(args.create_dataset,f'top_k_{data_type}_document.csv')
    #     top_k_document_df.to_csv(output_file_path, index=False)
    #     display(f"Saved top_k_document for {data_type} to {output_file_path}")



    process.end()
    process_main.end()
    

if __name__ == "__main__":
    main()
