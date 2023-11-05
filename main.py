from train_background import training,preprocessing,logging
import os
import os
import sys
from datetime import datetime
import time
import argparse
#import common
import pandas as pd
#from common import recorder
from ranking import log_process,load_data,display,init_model_and_tokenizer,create_dataloader,predict_scores,get_and_save_top_k_sentences_with_token_length,plot_and_save_boxplot
from ranking import calculate_rouge_for_each_review, cal_rouge
from torch import nn

#from generator_new_token import train
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

def generator_main():
    parser = argparse.ArgumentParser(description="Text Summarization Training")
    
    parser.add_argument('--log', action='store_true',help='logging')
    parser.add_argument('--log_dir', type=str, default="generating", help='Logging directory')
    parser.add_argument('--full', action='store_true', help='Run on full data')
    parser.add_argument('--train_file', type=str, help='Path to the training file', default='/home/Thesis/dataset-sample/Top_2_sentences_with_token_length_for_each_ReviewID_train.csv')
    parser.add_argument('--val_file', type=str, help='Path to the validation file', default='/home/Thesis/dataset-sample/Top_2_sentences_with_token_length_for_each_ReviewID_dev.csv')
    parser.add_argument('--model', type=str, default='bart_cnn')
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--n_epoch', type=int, default=3)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--exp_name', type=str,default ='PICO-masking')
    args = parser.parse_args()
    args.train_id = datetime.now().strftime("%Y%m%d-%H%M")

    train_df = pd.read_csv(args.train_file)
    val_df = pd.read_csv(args.val_file)
    
    MODEL_PATH = {
    'bart_base':"facebook/bart-base"}

    for model_name, model_path in MODEL_PATH.items():
        args.model = model_name
        # adjust logging to reflect the current model being trained
        log_dir = os.path.join('log', args.exp_name if args.exp_name else '', model_name)
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        logging.basicConfig(
            filename=f'{log_dir}/{args.train_id}.log',
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        display(f"Start training with {args} using model {model_name}")
        process = log_process(f"Training", log_=True, print_=True)
        display(f"Imported {len(train_df)} rows of train dataset")
        display(f"Imported {len(val_df)} rows of val dataset")
        display(f"Train and Validation Data loaded")

        train(args, train_df, val_df)
        process.end()


def ranking_main():
    # Initialize paths
    parser = argparse.ArgumentParser(description='Ranking')
    parser.add_argument('--log', action='store_true',
                        help='logging')

    parser.add_argument('--log_dir', type=str, default="pico_0.15/ranking",
                        help='logging directory')
    parser.add_argument('--full', action='store_true',
                        help='run full samples')
    parser.add_argument('--bs', type=int, default=64,
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
    
    for dtype in datatype:
        val_path = rf"/home/Thesis/dataset/PICO_0.15/{dtype}_PICO_0.15.csv"
  
        base_path = "/home/Thesis/pico_0.15/background_checkpoints/"
        subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

        if subdirs:
            # Get the latest created subdirectory
            latest_subdir = max(subdirs, key=lambda d: os.path.getctime(os.path.join(base_path, d)))
            checkpoint_path = os.path.join(base_path, latest_subdir)
        else:
            checkpoint_path = base_path
        display(checkpoint_path)
        display(f'checkpoint_path:{checkpoint_path}')
        save_path_dir = r"/home/Thesis/pico_0.15/ranking_pico_0.15_result"
        if not os.path.exists(save_path_dir): 
                os.makedirs(save_path_dir)
        result_path = r"/home/Thesis/pico_0.15/ranking_pico_0.15_result"
        if not os.path.exists(result_path): 
                os.makedirs(result_path)
        process_main = log_process(f"Program Running", log_=args.log, print_=True)
        process = log_process(f"Loading data", log_=args.log, print_=True)
        train_df = load_data(val_path)
        display(f"Imported {len(train_df)} rows of {dtype} datasset")

        tokenizer, model = init_model_and_tokenizer(checkpoint_path)
        display(f"tokenizer and model loded")

        dataloader = create_dataloader(train_df, tokenizer,batch_size=args.bs)
        display(f"Data loader created")

        predictions = predict_scores(dataloader, model)

        

        train_df['predicted_score'] = predictions    
        display(f"Predicted Score")

        get_and_save_top_k_sentences_with_token_length(train_df, [2], tokenizer,result_path,dtype)
        
    # Load these into DataFrames.
        dfs = [
            pd.read_csv(os.path.join(result_path, f"Top_{k}_sentences_with_token_length_for_each_ReviewID_{dtype}.csv")) 
            for k in [2]
        ]
        #box_save_path = os.path.join(save_path_dir,f'my_custom_boxplot{dtype}.png')
        #plot_and_save_boxplot(dfs, save_path=box_save_path)

        rouge_score = calculate_rouge_for_each_review(train_df)
        display(f"average rouge score of Extractive summary")
        for k, df in zip([2], dfs):
            rouge_scores = cal_rouge(df)
        #     rouge_scores_dict = {
        #         2 : cal_rouge(dfs[0]),
        #     }
        #     save_path_location = os.path.join(save_path_dir,f'rouge_score_measure{dtype}.png')
        #     plot_fmeasure_rouge(rouge_scores_dict,save_path_location)
        #     display(f"ROUGE scores for top {k} sentences:")
            
            for metric, score in rouge_scores.items():
                display(f"Overall {metric.upper()}: {score}")

    
    process.end()
    process_main.end()
def display(display_txt):
    logging.info(display_txt)
    print(display_txt)
def train_background_main():
    parser = argparse.ArgumentParser(description='Extracting')
    # Training settings
    parser.add_argument('--bs', type=int, default=192,
                        help='batch size (default: 16)')
    parser.add_argument('--n_epoch', type=int, default=1,
                        help='number of epochs (default: 5)')
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='learning rate')
    # Result path settings
    parser.add_argument('--model_dir', type=str, default="pico_0.3/background_checkpoint",
                        help='saved model directory')
    # File running settings
    parser.add_argument('--full', action='store_true',
                        help='run full samples')
    parser.add_argument('--log', action='store_true',
                        help='logging')
    parser.add_argument('--log_dir', type=str, default="pico_0.3/background_modeling",
                        help='logging directory')
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
    process_main = log_process(f"Main program", log_=args.log, print_=True)

    process = log_process(f"Loading data", log_=args.log, print_=True)
    train_df = preprocessing(r'/home/Thesis/dataset/PICO_0.3/train_PICO_0.3.csv')
    display(f"Imported {len(train_df)} rows of train datasset")
    val_df = preprocessing(r'/home/Thesis/dataset/PICO_0.3/dev_PICO_0.3.csv')
    display(f"Imported {len(val_df)} rows of validation dataset")
    process.end()

    process = log_process(f"Training", log_=args.log, print_=True)
    training(args, train_df, val_df)
    process.end()

    process_main.end()
    




if __name__ == '__main__':
    start_time = time.time()

    train_background_main()
    #ranking_main()

    end_time = time.time()
    total_seconds = end_time - start_time

    # Convert total_seconds into hours, minutes, and seconds
    hours = int(total_seconds // 3600)
    total_seconds %= 3600
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)

    print(f"Total execution time: {hours} hours, {minutes} minutes, and {seconds} seconds")