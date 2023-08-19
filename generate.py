import torch
import sys
import config
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from train.utils import generate_stopping_criteria_list
from eval.dialogs import PromptGenerator, STOP_WORDS

   

def main(model_dir: str, max_tokens: int):    
    transformers.logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
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
    try:
        model_dir = sys.argv[1]
        max_tokens = int(sys.argv[2])
    except:
        raise Exception("Corrupted call, for example: python generate.py target_model/ 100")
    
    main(model_dir, max_tokens)


