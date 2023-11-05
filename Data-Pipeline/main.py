from train_background import training,preprocessing,logging
import os
import os
import sys
from datetime import datetime
import argparse
import common
import pandas as pd
from common import recorder
from ranking import log_process,load_data,display,init_model_and_tokenizer,create_dataloader,predict_scores,get_and_save_top_k_sentences_with_token_length,plot_and_save_boxplot
from ranking import calculate_rouge_for_each_review, cal_rouge,plot_fmeasure_rouge
from torch import nn

from generator_new_token import train


def generator_main():
    parser = argparse.ArgumentParser(description="Text Summarization Training")
    
    parser.add_argument('--log', action='store_true',help='logging')
    parser.add_argument('--log_dir', type=str, default="generating", help='Logging directory')
    parser.add_argument('--full', action='store_true', help='Run on full data')
    parser.add_argument('--train_file', type=str, help='Path to the training file', default='/home/akoirala/Thesis/Data-Pipeline/random_0.15/trail_result/Top_2_sentences_with_token_length_for_each_ReviewID_train.csv')
    parser.add_argument('--val_file', type=str, help='Path to the validation file', default='/home/akoirala/Thesis/Data-Pipeline/random_0.15/trail_result/Top_2_sentences_with_token_length_for_each_ReviewID_dev.csv')
    parser.add_argument('--model', type=str, default='bart_cnn')
    parser.add_argument('--bs', type=int, default=12)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--n_epoch', type=int, default=3)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--exp_name', type=str,default ='PICO-trail')
    args = parser.parse_args()
    args.train_id = datetime.now().strftime("%Y%m%d-%H%M")

    train_df = pd.read_csv(args.train_file)
    val_df = pd.read_csv(args.val_file)
    
    MODEL_PATH = {
    'bart_base':"facebook/bart-base",
    'bart_large':"facebook/bart-large",
    'bart_cnn': "facebook/bart-large-cnn",
    'bart_xsum':"facebook/bart-large-xsum"}
    
    types = ['top_k_sentences', 'masking_background', 'bg_top_k_sentences']

    for model_name, model_path in MODEL_PATH.items():
        for type_ in types:
            args.model = model_name
            args.type = type_

            # adjust logging to reflect the current model and type being trained
            log_dir = os.path.join('log', args.exp_name if args.exp_name else '', model_name, type_)
            if not os.path.exists(log_dir): os.makedirs(log_dir)

            logging.basicConfig(
                filename=f'{log_dir}/{args.train_id}.log',
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            display(f"Start training with {args} using model {model_name} for type {type_}")
            process = log_process(f"Training", log_=True, print_=True)
            display(f"Imported {len(train_df)} rows of train dataset")
            display(f"Imported {len(val_df)} rows of val dataset")
            display(f"Train and Validation Data loaded")

            train(args, train_df, val_df, type_)
            process.end()
    
    # types = ['top_k_sentences','masking_background','bg_top_k_sentences']

    # for model_name, model_path in MODEL_PATH.items():
    #     args.model = model_name
    #     # adjust logging to reflect the current model being trained
    #     log_dir = os.path.join('log', args.exp_name if args.exp_name else '', model_name)
    #     if not os.path.exists(log_dir): os.makedirs(log_dir)
    #     logging.basicConfig(
    #         filename=f'{log_dir}/{args.train_id}.log',
    #         level=logging.INFO,
    #         format="%(asctime)s [%(levelname)s]: %(message)s",
    #         datefmt="%Y-%m-%d %H:%M:%S",
    #     )

    #     display(f"Start training with {args} using model {model_name}")
    #     process = log_process(f"Training", log_=True, print_=True)
    #     display(f"Imported {len(train_df)} rows of train dataset")
    #     display(f"Imported {len(val_df)} rows of val dataset")
    #     display(f"Train and Validation Data loaded")

    #     train(args, train_df, val_df,types)
    #     process.end()


def ranking_main():
    # Initialize paths
    parser = argparse.ArgumentParser(description='Ranking')
    parser.add_argument('--log', action='store_true',
                        help='logging')

    parser.add_argument('--log_dir', type=str, default="random_0.15/ranking",
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
    datatype=['train','dev'] #..............................change 
    for dtype in datatype:
        val_path = rf"/home/akoirala/Thesis/Data-Pipeline/random_dataset/random_0.15/top_6_{dtype}.csv"
        base_path = "/home/akoirala/Thesis/Data-Pipeline/random_0.15/background_checkpoints/"
        subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

        if subdirs:
            # Get the latest created subdirectory
            latest_subdir = max(subdirs, key=lambda d: os.path.getctime(os.path.join(base_path, d)))
            checkpoint_path = os.path.join(base_path, latest_subdir)
        else:
            checkpoint_path = base_path
        display(checkpoint_path)
        display(f'checkpoint_path:{checkpoint_path}')
        save_path_dir = r"/home/akoirala/Thesis/Data-Pipeline/random_0.15/trail_result"
        if not os.path.exists(save_path_dir): 
                os.makedirs(save_path_dir)
        result_path = r"/home/akoirala/Thesis/Data-Pipeline/random_0.15/trail_result"
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
            # rouge_scores_dict = {
            #     2 : cal_rouge(dfs[0])
            # }
            # save_path_location = os.path.join(save_path_dir,f'rouge_score_measure{dtype}.png')
            # plot_fmeasure_rouge(rouge_scores_dict,save_path_location)
            # display(f"ROUGE scores for top {k} sentences:")
            
            for metric, score in rouge_scores.items():
                display(f"Overall {metric.upper()}: {score}")

    
    process.end()
    process_main.end()

def train_background_main():
    parser = argparse.ArgumentParser(description='Extracting')
    # Training settings
    parser.add_argument('--bs', type=int, default=128,
                        help='batch size (default: 16)')
    parser.add_argument('--n_epoch', type=int, default=3,
                        help='number of epochs (default: 5)')
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='learning rate')
    # Result path settings
    parser.add_argument('--model_dir', type=str, default="background_checkpoint",
                        help='saved model directory')
    # File running settings
    parser.add_argument('--full', action='store_true',
                        help='run full samples')
    parser.add_argument('--log', action='store_true',
                        help='logging')
    parser.add_argument('--log_dir', type=str, default="random_0.15/background_modeling",
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
    common.display(f"{args=}")
    process_main = common.log_process(f"Main program", log_=args.log, print_=True)

    process = common.log_process(f"Loading data", log_=args.log, print_=True)
    train_df = preprocessing(r'/home/akoirala/Thesis/Data-Pipeline/random_dataset/random_0.15/top_6_train.csv')
    common.display(f"Imported {len(train_df)} rows of train datasset")
    val_df = preprocessing(r'/home/akoirala/Thesis/Data-Pipeline/random_dataset/random_0.15/top_6_dev.csv')
    common.display(f"Imported {len(train_df)} rows of train datasset")
    common.display(f"Imported {len(val_df)} rows of validation dataset")
    process.end()

    process = common.log_process(f"Training", log_=args.log, print_=True)
    training(args, train_df, val_df)
    process.end()

    process_main.end()
    




if __name__ == '__main__':
    train_background_main()
    ranking_main()
    generator_main()