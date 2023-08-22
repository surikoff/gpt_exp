import argparse
from config import TRAINING_CONFIG, BNB_CONFIG, LORA_CONFIG
from constants import TRAIN_MODE
from train.llm_trainer import LLMTrainer
from train.llm_trainer_lora import LLMTrainerLora


def main(data_file: str, dump_folder: str, num_train_epochs: int, mode: str, lora_weights: str = None):
    if mode == TRAIN_MODE.LORA:
        gpt_trainer = LLMTrainerLora(
            data_file, 
            dump_folder,
            bnb_config = BNB_CONFIG, 
            lora_config = LORA_CONFIG,
            lora_weights = lora_weights,
            **TRAINING_CONFIG,            
        )
        gpt_trainer.train(num_train_epochs)
    else:
        gpt_trainer = LLMTrainer(data_file, dump_folder, **TRAINING_CONFIG)
        gpt_trainer.train(num_train_epochs)
        
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
