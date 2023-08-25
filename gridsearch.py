import argparse
import datetime
from config import TRAINING_CONFIG, BNB_CONFIG, LORA_CONFIG
from constants import TRAIN_MODE
from train import LLMTrainer, LLMTrainerLora
from utils import duration_printer


def main(data_file: str, dump_folder: str, num_train_epochs: int, mode: str, lora_weights: str = None):
    start_time = datetime.datetime.now()
    if mode == TRAIN_MODE.LORA:        
        for r in [1,4,16,32,64]:
            for a in [4,16,32]:
                print(f"Trainer set to Lora mode r={r} a={a}")
                conf = dict(LORA_CONFIG)
                conf["r"] = r
                conf["lora_alpha"] = a
                gpt_trainer = LLMTrainerLora(
                    data_file, 
                    f"{dump_folder}_r{r}_a{a}",
                    bnb_config = BNB_CONFIG, 
                    lora_config = conf,
                    lora_weights = lora_weights,
                    **TRAINING_CONFIG,            
                )
                gpt_trainer.train(num_train_epochs)
    else:
        gpt_trainer = LLMTrainer(data_file, dump_folder, **TRAINING_CONFIG)
        gpt_trainer.train(num_train_epochs)
    finish_time = datetime.datetime.now()     
    # gpt_trainer.save_model()
    duration_printer((finish_time-start_time).total_seconds())


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
