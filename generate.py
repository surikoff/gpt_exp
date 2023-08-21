import torch
import argparse
import transformers
from eval.dialogs import PromptGenerator
from config import BASE_MODEL_DIR, BNB_CONFIG, EVAL_CONFIG
from constants import EVAL_MODE
from eval.generator import LLMGenerator
from eval.generator_bnb import LLMGeneratorBNB


def main(model_dir: str, mode: str):
    prompt_generator = PromptGenerator()
    if mode == EVAL_MODE.BNB:
        llm_generator = LLMGeneratorBNB(
            model_dir, 
            bnb_config = BNB_CONFIG,
            **EVAL_CONFIG)
    elif mode == EVAL_MODE.LORA:
    else:
        llm_generator = LLMGenerator(model_dir, **EVAL_CONFIG)
    while True:
        text = input("Input: ")
        prompt_generator.request(text)
        # print("!\n", prompt_generator.prompt, "!\n")
        response = llm_generator.generate(prompt_generator.prompt)        
        prompt_generator.response(response)
        print(response)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for generating dialogs with the LLM model")
    parser.add_argument(
        'model_dir', 
        type=str, 
        help='Target LLM model folder')
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=list(EVAL_MODE.__dict__.values()), 
        required=False, 
        default=EVAL_MODE.NATIVE, 
        help='Evaluation mode, default: native')
    args = parser.parse_args()    
    
    main(args.model_dir, args.mode)



