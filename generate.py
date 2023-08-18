import torch
import sys
import os.path
from transformers import AutoTokenizer, AutoModelForCausalLM


def main(model_dir: str, max_tokens: int, prompt: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokens = tokenizer(prompt).input_ids
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    output = model.generate(
        torch.tensor([tokens]).to(model.device),
        max_new_tokens=max_tokens
    )[0]

    print(tokenizer.decode(output))


if __name__ == '__main__':
    try:
        model_dir = sys.argv[1]
        max_tokens = int(sys.argv[2])
        prompt = sys.argv[3]
    except:
        raise Exception("Corrupted call, for example: python generate.py target_model/ 100 'Раз, два, три, четыре, пять'")
    
    main(model_dir, max_tokens, prompt)
