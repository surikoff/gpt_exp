import torch
import argparse
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftConfig, PeftModel
from train.utils import generate_stopping_criteria_list
from eval.dialogs import PromptGenerator, STOP_WORDS
from config import BASE_MODEL_DIR, BNB_CONFIG, EVAL_CONFIG
from constants import TRAIN_MODE
from eval.generate import LLMGenerator


def main(model_dir: str, max_tokens: int, mode: str):
    transformers.logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    
    # if mode == "bnb":
    #     model = AutoModelForCausalLM.from_pretrained(
    #         BASE_MODEL_DIR, 
    #         quantization_config =  BitsAndBytesConfig(**BNB_CONFIG), 
    #         torch_dtype = torch.float16            
    #     )
    # elif mode =="lora":
    #     config = PeftConfig.from_pretrained(model_dir)           
    #     model = AutoModelForCausalLM.from_pretrained(
    #         config.base_model_name_or_path, 
    #         quantization_config =  BitsAndBytesConfig(**BNB_CONFIG), 
    #         torch_dtype = torch.float16            
    #     )
    #     model = PeftModel.from_pretrained(model, model_dir)
    # else:
    #     model = AutoModelForCausalLM.from_pretrained(model_dir)

    
        
    prompt_generator = PromptGenerator(tokenizer)
    llm_generator = LLMGenerator(model_dir, **EVAL_CONFIG)

    

    while True:
        text = input("Input: ")
        prompt_generator.request(text)
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
        '--max_tokens', 
        type=int, 
        required=False, 
        default=100, 
        help='Maximum tokens to generation')    
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=list(TRAIN_MODE.__dict__.values()), 
        required=False, 
        default=TRAIN_MODE.NATIVE, 
        help='Evaluation mode, default: native')
    args = parser.parse_args()    
    
    main(args.model_dir, args.max_tokens, args.mode)


