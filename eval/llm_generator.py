import os
import torch
from torch import LongTensor
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from argparse import Namespace
from train.utils import generate_stopping_criteria_list


DEBUG_MODE = False
# The minimum number of tokens in the context window to accommodate the prompt tail
MIN_PROMPT_WINDOW_LENGTH = 100
# Generation parameters
REPETITION_PENALITY = 1.2


class LLMGenerator:
    def __init__(self, model_dir: str, **kwargs):
        self.config = Namespace(**kwargs)        
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        transformers.logging.set_verbosity_error()
        self._max_input_len = self.config.context_size - self.config.max_tokens
        if self._max_input_len < MIN_PROMPT_WINDOW_LENGTH:
            raise Exception(f"Context size {self.config.context_size} too low to generate with max length {self.config.max_tokens}")

        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._stopping_criteria_list = generate_stopping_criteria_list(
            self._tokenizer, 
            self.config.stop_words)
        self._model = self._init_model(model_dir)
        self._generation_config = GenerationConfig(
            max_new_tokens = self.config.max_tokens,
            repetition_penalty = REPETITION_PENALITY,
            pad_token_id = self._tokenizer.eos_token_id,
            # num_beams=2,
            # do_sample=True
        )
        

    def _init_model(self, model_dir: str) -> AutoModelForCausalLM:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype = self.config.data_type,
            device_map = "auto"
        )
        return model
        

    def generate(self, prompt: str) -> str:        
        input_ids = self._tokenize(prompt)[-self._max_input_len:]
        if DEBUG_MODE:
            print(f"\n\nPrompt:\n[{self._tokenizer.decode(input_ids)}]\n\n")        
        output_ids = self._model.generate(
            inputs = LongTensor([input_ids]).to(self._model.device),
            generation_config = self._generation_config,
            stopping_criteria = self._stopping_criteria_list
        )[0]
        output_text = self._decode(output_ids[len(input_ids):])
        return self._clear_output_text(output_text)
    
    def _clear_output_text(self, text):
        text = text.strip()
        if text[-1] in self.config.stop_words:
            text = text[:-1]
        return text

    def _tokenize(self, text: str) -> list:
        return self._tokenizer(text).input_ids
    
    def _decode(self, text: str) -> list:
        return self._tokenizer.decode(text)
    
        