import argparse
import datetime
import os
import wandb
from config import TRAINING_CONFIG, BNB_CONFIG, LORA_CONFIG, WANDB_PROJECT, WANDB_KEY
from constants import TRAIN_MODE
from train import LLMTrainer, LLMTrainerLora
from utils import duration


def main(data_file: str, dump_folder: str, num_train_epochs: int, mode: str, lora_weights: str = None):
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    os.environ["WANDB_LOG_MODEL"] = "all"
    wandb.login(key=WANDB_KEY)

    start_time = datetime.datetime.now()
    print("Loading model from", TRAINING_CONFIG["model_path"])
    if mode == TRAIN_MODE.LORA:
        print("Trainer set to Lora mode...")
        gpt_trainer = LLMTrainerLora(
            data_file, 
            dump_folder,
            bnb_config = BNB_CONFIG, 
            lora_config = LORA_CONFIG,
            lora_weights = lora_weights,
            **TRAINING_CONFIG,            
        )
    else:
        gpt_trainer = LLMTrainer(data_file, dump_folder, **TRAINING_CONFIG)
    finish_time = datetime.datetime.now()     
    print("Model loaded in", duration((finish_time-start_time).total_seconds()))
    
    start_time = datetime.datetime.now()
    gpt_trainer.train(num_train_epochs)
    finish_time = datetime.datetime.now()
    print("Model has been trained in", duration((finish_time-start_time).total_seconds()))
    wandb.finish()
    gpt_trainer.save_model()


if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description="Script for training LLM models")
    parser.add_argument('data', type=str, help='Books data file path')
    parser.add_argument('output', type=str, help='Output dir for saving finetuned model')
    parser.add_argument('--epochs', type=int, required=False, default=1, help='Number of training epochs')
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=list(TRAIN_MODE.__dict__.values()), 
        required=False, 
        default=TRAIN_MODE.NATIVE, 
        help='Training mode, default: native')
    parser.add_argument('--lora_weights', type=str, required=False, default=None, help='Location of pretrained weights for Lora mode')
    args = parser.parse_args()    
    if args.lora_weights is not None and args.mode is not TRAIN_MODE.LORA:
        args.mode = TRAIN_MODE.LORA

    main(args.data, args.output, args.epochs, args.mode, args.lora_weights)
