import torch
from torch import LongTensor
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from argparse import Namespace
from train.utils import generate_stopping_criteria_list



class LLMGenerator:
    def __init__(self, model_dir: str, **kwargs):
        self.config = Namespace(**kwargs)        
        # transformers.logging.set_verbosity_error()
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_dir, 
            pad_token_id = self._tokenizer.eos_token_id).to(self.config.device)
        self._stopping_criteria_list = generate_stopping_criteria_list(
            self._tokenizer, 
            self.config.stop_words)

    def generate(self, prompt: str) -> str:
        max_input_len = self.config.context_size - self.config.max_tokens
        input_ids = self._tokenize_prompt(prompt)
        input_ids = input_ids[-max_input_len:]
        print(f"\n\n!{self._tokenizer.decode(input_ids)}!\n\n")
        output_ids = self._model.generate(
            inputs = LongTensor([input_ids], device=self.config.device),
            max_new_tokens = self.config.max_tokens,
            num_beams=self.config.num_beams,
            do_sample=self.config.do_sample,
            stopping_criteria = self._stopping_criteria_list
        )[0]
        output_text = self._tokenizer.decode(output_ids[len(input_ids):])
        return output_text.strip()

    
    def _tokenize_prompt(self, prompt: str) -> list:
        return self._tokenizer(prompt).input_ids
    
        