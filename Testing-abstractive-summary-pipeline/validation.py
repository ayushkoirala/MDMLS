
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import torch
import time
import pandas as pd
import numpy as np
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import AutoTokenizer, AutoModelForTokenClassification
from nltk.tokenize import sent_tokenize
import logging
import argparse
from tqdm import tqdm
import random
import os
from torch.utils.data import DataLoader, TensorDataset
from rouge import Rouge
from datetime import datetime
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer
from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data import DataLoader, Dataset
from bert_score import score as bert_score
from bart_score import BARTScorer
from rouge_score import rouge_scorer

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
def top_k_document(sample_train_inputs, top_k, device):
    # Initialize the context encoder and tokenizer
    context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    
    # Initialize the question encoder and tokenizer for encoding the background
    background_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    background_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

    # Move models to the specified device
    context_encoder.to(device)
    background_encoder.to(device)

    timings = {}
    desired_output = []

    batch_size = 5

    for review_id in sample_train_inputs['ReviewID'].unique():
        start_time = time.time()
        review_entries = sample_train_inputs[sample_train_inputs['ReviewID'] == review_id]
        review_abstracts = review_entries['Abstract'].tolist()
        background_data = review_entries['Target'].tolist()[0]

        retrieval_scores_list = []
        batch_abstracts = []

        for abstract in review_abstracts:
            batch_abstracts.append(abstract)
            if len(batch_abstracts) == batch_size:
                # Tokenize and encode the abstracts using the context encoder
                abstract_tokens = context_tokenizer(batch_abstracts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                multi_document_embeddings = context_encoder(**abstract_tokens).pooler_output
                
                # Tokenize and encode the background using the background encoder
                background_tokens = background_tokenizer([background_data], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                background_embeddings = background_encoder(**background_tokens).pooler_output

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
                'ground_truth_summary': top_k_candidate_target
            })

    desired_output_df = pd.DataFrame(desired_output)
    return desired_output_df

def remove_hash_concat(row):
    def remove_unwanted_chars(text):
        if isinstance(text, str):
            cleaned_text = text.replace('#', '').replace('\n', '')
            return cleaned_text
        return text

    for col in ['ground_truth_summary','docs_sent','input_seq']:
        row[col] = remove_unwanted_chars(row[col])
    return row

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

def init_model_and_tokenizer(checkpoint):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=1)
    return tokenizer, model

def get_and_save_top_k_sentences_with_token_length(val_df, k_values, tokenizer,result_path):
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
        excel_name = os.path.join(result_path, f"Top_{k}_sentences_with_token_length_for_each_ReviewID.csv")
        top_k_df.to_csv(excel_name, index=False)
        return top_k_df

def preprocess_background(background, tokenizer, model, device,mask_ratio):
    background = background.replace("(", "").replace(")", "").replace(",", "").replace("\n", " ")
    inputs = tokenizer.encode(background, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    outputs = model(inputs).logits
    predictions = outputs.argmax(dim=2)[0]

    tokens = tokenizer.convert_ids_to_tokens(inputs[0])
    masked_tokens = []
    for token, prediction in zip(tokens, predictions):
        label = model.config.id2label[prediction.item()]

        if label != 'O' and random.random() < mask_ratio:
            token = tokenizer.mask_token
            
        if token not in ['[CLS]', '[SEP]']:
            masked_tokens.append(token)

    return " ".join(masked_tokens)

class MyDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.inputs = tokenizer(list(data['top_k_sentences']), padding='max_length', truncation=True, return_tensors='pt')
        self.targets = tokenizer(list(data['ground_truth_summary']), padding='max_length', truncation=True, return_tensors='pt')

    def __getitem__(self, index):
        items = {
            'input_ids': self.inputs['input_ids'][index],
            'attention_mask': self.inputs['attention_mask'][index],
            'labels': self.targets['input_ids'][index],
            # 'output_attentions': self.targets['attention_mask'][index]
        }
        return items

    def __len__(self):
        return len(self.data)

def build_loader(data, tokenizer, bs,split= 'val'):
    build_process = log_process(f"Buliding val dataset", log_=True, print_=True)
    # Build dataset
    process = log_process(f"Buliding MyDataset", log_=True, print_=True)
    dataset = MyDataset(data, tokenizer)
    process.end()
    # Bulid dataloader
    process = log_process(f"Buliding DataLoader", log_=True, print_=True)
    data_loader = DataLoader(dataset, batch_size=bs, shuffle=(split=='train'))
    process.end()
    build_process.end()
    return data_loader

def display(display_txt):
    logging.info(display_txt)
    print(display_txt)

# class recorder:
#     def __init__(self, main_folder="records"):
#         self.save_dir = main_folder
#         self.record_dict = {}
    
#     def record(self, key, val):
#         if key not in self.record_dict.keys(): 
#             self.record_dict[key] = [val]
#         else: 
#             self.record_dict[key].append(val)

#     def save(self):
#         if not os.path.exists(self.save_dir): 
#             os.makedirs(self.save_dir)
#         for key, val in self.record_dict.items():
#             filepath = os.path.join(self.save_dir, f"{key}.npy")
#             np.save(filepath, np.array(val))
#             display(f"Saved {key} to {filepath}")

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

def cal_bartscore(pred, ref, mode='avg', device='cpu'):
    bart_scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
    scores = bart_scorer.score(pred, ref)
    if mode=='avg':
        return sum(scores)/len(scores)
    else:
        return scores

def cal_matrix(pred, ref, mode='avg', device='cpu', metrix=['rouge', 'bertscore', 'bartscore']):
    process = log_process('Calculating Rouge score', print_=True)
    if 'rouge' in metrix: eval_score = cal_rouge(pred, ref, mode)
    process.end()
    process = log_process('Calculating BERT score', print_=True)
    process.end()
    if 'bertscore' in metrix: eval_score['BERTscore'] = cal_bertscore(pred, ref, mode)
    process = log_process('Calculating BART score', print_=True)
    if 'bartscore' in metrix: eval_score['BARTscore'] = cal_bartscore(pred, ref, mode, device)
    process.end()
    return eval_score

class recorder:
    def __init__(self, main_folder="records", subfolder='rec', train_id=None):
        if train_id:
            self.save_dir = f"{main_folder}/{subfolder}/{train_id}"
        else:
            if not os.path.exists(main_folder): 
                os.makedirs(main_folder)
                max_checkpoint = 0
            else:
                existing_folders = [folder for folder in os.listdir(main_folder) if folder.startswith(f'{subfolder}-')]
                max_checkpoint = max([int(folder.split('-')[1]) for folder in existing_folders], default=0)
            self.save_dir = f"{main_folder}/{subfolder}-{max_checkpoint+1}"
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        self.record_dict = {}

    def record(self, key, val):
        if key not in self.record_dict.keys(): self.record_dict[key] = [val]
        else: self.record_dict[key].append(val)

    def save(self):
        # if sub_folder: main_folder += f"/{sub_folder}"
        # if train_id: main_folder += f"/{train_id}"
        for key, val in self.record_dict.items():
            filepath = f"{self.save_dir}/{key}.npy"
            np.save(filepath, np.array(val))
            display(f"Saved {key} to {filepath}")

def validate(args, val_data,device,val_exp):
    
    #tokenizer = BartTokenizer.from_pretrained(args.bart_checkpoint)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    n_epoch = 3
    
    val_loader = build_loader(val_data, split='val', tokenizer=tokenizer, bs=args.bs)
    val_records = recorder()
    sumgen_df = val_data.copy()
    if args.save_samp and args.split[:4]!='test': 
        if args.full: random_samples = random.sample(range(0, len(sumgen_df)), int(len(sumgen_df)*0.01))
        else: random_samples = random.sample(range(0, len(sumgen_df)), int(len(sumgen_df)*0.5))
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    
    for epoch in range(n_epoch):
        model = BartForConditionalGeneration.from_pretrained(args.bart_checkpoint)
        model.to(device)
        with torch.no_grad():
            gen_seq_list, ref_seq_list = [], []
            for batch in tqdm(val_loader, desc=f"[Epoch {epoch+1}] Validating"):
                for key in batch.keys(): batch[key] = batch[key].to(device)
                generated_ids = model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_length=tokenizer.model_max_length, 
                    num_beams=args.num_beams
                )
                gen_seq_list.extend([tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids])
                ref_seq_list.extend([tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['labels']])

            if args.save_samp and args.split[:4]!='test':
                sumgen_df[f'gen_seq_ep{epoch}'] = gen_seq_list
                filepath = f"/home/akoirala/Thesis/Testing/result/model_with_test/val_samples.csv"
                sumgen_df.iloc[random_samples].to_csv(filepath)
                display(f"Saved {len(sumgen_df.iloc[random_samples])} val samples to {filepath}")
            if args.split[:4]=='test':
                sumgen_df[f'gen_seq_ep{epoch}'] = gen_seq_list
                filepath = f"/home/akoirala/Thesis/Testing/result/model_with_test/generated_blindtest.csv"
                sumgen_df.to_csv(filepath)
                display(f"Saved {len(sumgen_df)} val samples to {filepath}")
            else:
                val_scores = cal_matrix(pred=gen_seq_list, ref=ref_seq_list, device=device)
                for key, val in val_scores.items(): val_records.record(key, val)
        val_records.save()
    
def create_dataloader(df, tokenizer, batch_size):
    df.dropna(subset=['input_seq'])
    df = df[df['input_seq'].apply(lambda x: isinstance(x, str))]
    inputs = tokenizer(list(df['input_seq']), padding="max_length", truncation=True, return_tensors="pt")
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], torch.tensor(df['final_score']))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader
def preprocess_background(background, tokenizer, model, device,mask_ratio):
    background = background.replace("(", "").replace(")", "").replace(",", "").replace("\n", " ")
    inputs = tokenizer.encode(background, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    outputs = model(inputs).logits
    predictions = outputs.argmax(dim=2)[0]

    tokens = tokenizer.convert_ids_to_tokens(inputs[0])
    masked_tokens = []
    for token, prediction in zip(tokens, predictions):
        label = model.config.id2label[prediction.item()]

        if label != 'O' and random.random() < mask_ratio:
            token = tokenizer.mask_token
            
        if token not in ['[CLS]', '[SEP]']:
            masked_tokens.append(token)

    return " ".join(masked_tokens)
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
                    'ground_truth_summary': row['ground_truth_summary'],
                    'input_seq': input_value  
                }
                result_df1.append(new_row)
    return pd.DataFrame(result_df1).reset_index(drop=True)

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
def main():
    parser = argparse.ArgumentParser(description="Text Summarization Testing")
    
    parser.add_argument('--log', action='store_true',help='logging')
    parser.add_argument('--log_dir', type=str, default="generating", help='Logging directory')
    parser.add_argument('--full', action='store_true', help='Run on full data')
    parser.add_argument('--test_file', type=str, help='Path to the training file', default='/home/akoirala/Thesis/Data-Preprocessing-Pipeline/test.csv')
    parser.add_argument('--background_checkpoint', type=str, help='Path to the training file', default='/home/akoirala/Thesis/training_query_model/checkpoint/model-with-test')
    parser.add_argument('--result_path', type=str, help='Path to the training file', default='/home/akoirala/Thesis/Testing/result/model_with_test')
    parser.add_argument('--model', type=str, default='bart_cnn')
    parser.add_argument('--bart_checkpoint', type=str, help='Path to the training file', default='/home/akoirala/Thesis/Generating_summary/checkpoints/checkpoint-12')
    parser.add_argument('--save_samp', action='store_true', help='save some samples of val input, target, generation')
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--split', type=str, default='sample')
    #parser.add_argument('--exp_name', type=str,default ='PICO-0.45-6-cnn-Top2-result')
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
    # input_df = pd.read_csv(args.test_file)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # k=6
    # top_k_document_df = top_k_document(input_df, k, device)
    # tokenizer = AutoTokenizer.from_pretrained("kamalkraj/BioELECTRA-PICO")
    # model = AutoModelForTokenClassification.from_pretrained("kamalkraj/BioELECTRA-PICO").to(device)

    # top_k_document_df['Masked Background'] = top_k_document_df['Background'].apply(preprocess_background, args=(tokenizer, model, device,0.15))
    # df_exploded = preprocess_dataframe(top_k_document_df)
    # rouge_scores_df = get_rouge_scores(df_exploded)
    # display("Getting Target R1,R2 and RL")
    
    # rouge_scores_df['final_score'] = rouge_scores_df.apply(calculate_final_score, axis=1)
    # display("Final Score is saved")

    # rouge_scores_df.drop(['R1_r', 'R2_r', 'R1_p', 'R2_p', 'Rl_p', 'Rl_r'], axis=1, inplace=True)
    # rouge_scores_df.to_csv(os.path.join(args.result_path,'PICO_0.15_test.csv'), index=False)
    # display("Process_data function Done")
    rouge_scores_df = pd.read_csv(r'/home/akoirala/Thesis/Testing/result/PICO_0.15_test.csv')
    #rouge_scores_df = remove_hash_concat(rouge_scores_df)
    rouge_scores_df_clean = rouge_scores_df.apply(remove_hash_concat, axis=1)
    tokenizer, model = init_model_and_tokenizer(args.background_checkpoint)
    dataloader = create_dataloader(rouge_scores_df_clean, tokenizer,batch_size=args.bs)
    display(f"Data loader created")

    predictions = predict_scores(dataloader, model)



    rouge_scores_df_clean['predicted_score'] = predictions    
    display(f"Predicted Score")

    device = torch.device('cuda') 
    rouge_scores_df_clean = get_and_save_top_k_sentences_with_token_length(rouge_scores_df_clean, [3], tokenizer,args.result_path)
    #top_k_df = pd.read_csv(r'/home/akoirala/Thesis/Testing/result/Top_3_sentences_with_token_length_for_each_ReviewID.csv')
    validate(args,rouge_scores_df_clean,device,'title')

if __name__ == "__main__":
    main()