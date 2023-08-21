import torch
from torch import LongTensor
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from argparse import Namespace
from train.utils import generate_stopping_criteria_list



class LLMGenerator:
    def __init__(self, model_dir: str, **kwargs):
        self.config = Namespace(**kwargs)        
        transformers.logging.set_verbosity_error()
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._stopping_criteria_list = generate_stopping_criteria_list(
            self._tokenizer, 
            self.config.stop_words)
        self._model = self._init_model(model_dir)

    def _init_model(self, model_dir: str) -> AutoModelForCausalLM:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype = self.config.data_type,
            device_map = "auto"
        )
        return model
        

    def generate(self, prompt: str) -> str:
        max_input_len = self.config.context_size - self.config.max_tokens
        input_ids = self._tokenize_prompt(prompt)
        input_ids = input_ids[-max_input_len:]
        print(f"\n\n!{self._tokenizer.decode(input_ids)}!\n\n")
        generation_config = GenerationConfig(
            max_new_tokens = self.config.max_tokens,
            repetition_penalty = 1.2,
            pad_token_id = self._tokenizer.eos_token_id,
            num_beams=2,
            do_sample=True
        )
        output_ids = self._model.generate(
            inputs = LongTensor([input_ids]).to(self._model.device),
            generation_config = generation_config,
            stopping_criteria = self._stopping_criteria_list
        )[0]
        output_text = self._tokenizer.decode(output_ids[len(input_ids):])
        return output_text.strip()

    
    def _tokenize_prompt(self, prompt: str) -> list:
        return self._tokenizer(prompt).input_ids
    
        