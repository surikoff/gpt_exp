import torch
import argparse
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from train.utils import generate_stopping_criteria_list
from eval.dialogs import PromptGenerator, STOP_WORDS
from config import BNB_CONFIG


def main(model_dir: str, max_tokens: int, mode: str):
    transformers.logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    
    if mode == "bnb":
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, 
            quantization_config =  BitsAndBytesConfig(**BNB_CONFIG), 
            torch_dtype = torch.float16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        
    prompt_generator = PromptGenerator(tokenizer)
    stopping_criteria_list = generate_stopping_criteria_list(tokenizer, STOP_WORDS)

    while True:
        text = input("Input: ")
        prompt_generator.request(text)
        output = model.generate(
            torch.tensor([prompt_generator.prompt_ids]).to(model.device),
            max_new_tokens = max_tokens,
            stopping_criteria = stopping_criteria_list
        )[0]
        response = prompt_generator.get_response(output)
        print(response)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for generating dialogs with the LLM model")
    parser.add_argument('model_dir', type=str, help='Target LLM model folder')
    parser.add_argument('--max_tokens', type=int, required=False, default=100, help='Maximum tokens to generation')
    parser.add_argument('--mode', type=str, choices=["native", "bnb"], required=False, default="native", help='Evaluation mode, default: native')
    args = parser.parse_args()    
    
    main(args.model_dir, args.max_tokens, args.mode)


