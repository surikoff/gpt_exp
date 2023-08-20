import argparse
from config import TRAINING_CONFIG, BNB_CONFIG, LORA_CONFIG
from train.gpt_trainer import GptTrainer
from train.gpt_trainer_lora import GptTrainerLora


def main(data_file: str, dump_folder: str, num_train_epochs: int, mode: str):
    if mode == "lora":
        gpt_trainer = GptTrainerLora(
            data_file, 
            bnb_config = BNB_CONFIG, 
            lora_config = LORA_CONFIG,
            **TRAINING_CONFIG,            
        )
        gpt_trainer.train(num_train_epochs)
    else:
        gpt_trainer = GptTrainer(data_file, **TRAINING_CONFIG)
        gpt_trainer.train(num_train_epochs)
        
    gpt_trainer.save_model(dump_folder)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for training LLM models")
    parser.add_argument('data', type=str, help='Books data file path')
    parser.add_argument('output', type=str, help='Output dir for saving finetuned model')
    parser.add_argument('--epochs', type=int, required=False, default=1, help='Number of training epochs')
    parser.add_argument('--mode', type=str, choices=["native", "lora"], required=False, default="native", help='Training mode, default: native')
    args = parser.parse_args()    
    
    main(args.data, args.output, args.epochs, args.mode)
