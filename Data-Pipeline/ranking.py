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
    df = pd.read_csv(path)
    df['ground_truth_summary'] = df['ground_truth_summary'].str.replace('\n', '')        
    return df

def init_model_and_tokenizer(checkpoint):
    display(checkpoint)
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
            background = " ".join(sorted_group['Background'])
            masked_background = background = " ".join(sorted_group['Masked Background'])
            first_target = group['ground_truth_summary'].iloc[0]
            
            # Calculate BART_token_len for this concatenated_sentences
            bart_token_len = len(tokenizer.encode(concatenated_sentences))
            
            results.append({
                'ReviewID': name,
                'sent_id': concatenated_sent_id,
                'top_k_sentences': concatenated_sentences,
                'predicted_score': concatenated_predicted_score,
                'ground_truth_summary': first_target,
                'BART_token_len': bart_token_len,
                'Background':background,
                'Masked Background':masked_background# Adding BART_token_len here
            })

        top_k_df = pd.DataFrame(results)
        excel_name = os.path.join(result_path, f"Top_{k}_sentences_with_token_length_for_each_ReviewID_{dtype}.csv")
        top_k_df.to_csv(excel_name, index=False)

def plot_and_save_boxplot(dfs, save_path):
    """
    Plots and saves a boxplot showing the distribution of token lengths for different top_k values.
    
    Parameters:
    - dfs: list of pd.DataFrame
        List of dataframes containing the token length data.
    - save_path: str
        The path where the boxplot will be saved.

    """
    #1,2 3, 5, 7, 10, 15, 20, 25
    # Combine dataframes
    dfs[0]['top_k'] = 2
    dfs[1]['top_k'] = 5
    dfs[2]['top_k'] = 10

   
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


def plot_fmeasure_rouge(scores_dict,save_path):
    """
    Plots the F-measure of ROUGE-1, ROUGE-2, and ROUGE-L scores for different top-k sentences.
    
    Parameters:
    - scores_dict: dict
        Dictionary containing the ROUGE scores for different top-k sentences.
        The keys should be the top-k values, and the values should be the corresponding ROUGE scores.
    """
    # Extract top-k values and corresponding F-measure scores for ROUGE-1, ROUGE-2, and ROUGE-L
    top_k_values = list(scores_dict.keys())
    fmeasure_scores_rouge1 = [scores_dict[k]['rouge1']['f'] for k in top_k_values]
    fmeasure_scores_rouge2 = [scores_dict[k]['rouge2']['f'] for k in top_k_values]
    fmeasure_scores_rougeL = [scores_dict[k]['rougeL']['f'] for k in top_k_values]
    
    # Create the line plot
    plt.figure(figsize=(10, 6))
    plt.plot(top_k_values, fmeasure_scores_rouge1, marker='o', label='ROUGE-1 F-measure')
    plt.plot(top_k_values, fmeasure_scores_rouge2, marker='s', label='ROUGE-2 F-measure')
    plt.plot(top_k_values, fmeasure_scores_rougeL, marker='^', label='ROUGE-L F-measure')
    plt.xlabel('Top-k Sentences')
    plt.ylabel('F-measure')
    plt.title('ROUGE F-measure for Different Top-k Sentences')
    plt.xticks(top_k_values)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()


def main():
    # Initialize paths
    parser = argparse.ArgumentParser(description='Ranking')
    parser.add_argument('--log', action='store_true',
                        help='logging')

    parser.add_argument('--log_dir', type=str, default="ranking",
                        help='logging directory')
    parser.add_argument('--full', action='store_true',
                        help='run full samples')
    parser.add_argument('--bs', type=int, default=32,
                        help='batch size (default: 32)')
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
    datatype=['train','dev']
    process_main = log_process(f"Program Running", log_=args.log, print_=True)
    process = log_process(f"Loading data", log_=args.log, print_=True)
    for dtype in datatype:
        val_path = rf"/home/akoirala/Thesis/Data-Preprocessing-Pipeline/result/PICO-0.15/PICO_0.15_{dtype}.csv"
        #checkpoint_path = r"/home/akoirala/Thesis/Final-pipeline/background_checkpoint/"
        # List all subdirectories
        subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join("/home/akoirala/Thesis/Final-pipeline/background_checkpoint/", d))]

        # Get the latest created subdirectory
        latest_subdir = max(subdirs, key=lambda d: os.path.getctime(os.path.join(base_path, d)))
        display(latest_subdir)
        checkpoint_path = os.path.join(base_path, latest_subdir)
        display(checkpoint_path)
        display(f'checkpoint_path:{checkpoint_path}')
        save_path_dir = r"/home/akoirala/Thesis/Final-pipeline/ranking_pico_0.15_result"
        result_path = r"/home/akoirala/Thesis/Final-pipeline/ranking_pico_0.15_result"
        
        train_df = load_data(val_path)
        display(f"Imported {len(train_df)} rows of {dtype} datasset")

        tokenizer, model = init_model_and_tokenizer(checkpoint_path)
        display(f"tokenizer and model loded")

        dataloader = create_dataloader(train_df, tokenizer,batch_size=args.bs)
        display(f"Data loader created")

        predictions = predict_scores(dataloader, model)

        

        train_df['predicted_score'] = predictions    
        display(f"Predicted Score")

        get_and_save_top_k_sentences_with_token_length(train_df, [2,5,10], tokenizer,result_path)
        
    # Load these into DataFrames.
        dfs = [
            pd.read_csv(os.path.join(result_path, f"Top_{k}_sentences_with_token_length_for_each_ReviewID.csv")) 
            for k in [2,5,10]
        ]
        box_save_path = os.path.join(save_path_dir,f'my_custom_boxplot{dtype}.png')
        plot_and_save_boxplot(dfs, save_path=box_save_path)

        rouge_score = calculate_rouge_for_each_review(train_df)
        display(f"average rouge score of Extractive summary")
        for k, df in zip([2], dfs):
            rouge_scores = cal_rouge(df)
            rouge_scores_dict = {
                2 : cal_rouge(dfs[0]),
                5 : cal_rouge(dfs[1]),
                10: cal_rouge(dfs[2])
            }
            save_path_location = os.path.join(save_path_dir,f'rouge_score_measure{dtype}.png')
            plot_fmeasure_rouge(rouge_scores_dict,save_path_location)
            display(f"ROUGE scores for top {k} sentences:")
            
            for metric, score in rouge_scores.items():
                display(f"Overall {metric.upper()}: {score}")

    
    process.end()
    process_main.end()
    

if __name__ == "__main__":
    main()
