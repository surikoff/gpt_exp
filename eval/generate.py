from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM




class LLMGenerator:
    def __init__(self, model_dir: str, **kwargs):
        self.config = Namespace(**kwargs)        
        transformers.logging.set_verbosity_error()
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._model = AutoModelForCausalLM.from_pretrained(model_dir)
        self._stopping_criteria_list = generate_stopping_criteria_list(
            self._tokenizer, 
            self.config.stop_words)


    def generate(self, prompt: str):
        output = model.generate(
            inputs = Tensor([prompt_generator.prompt_ids]).to(self._model.device),
            max_new_tokens = self.config.max_tokens,
            stopping_criteria = self._stopping_criteria_list
        )[0]
        output_text = self._tokenizer.decode(output)     
        response_len = len(output) - len(prompt)
        response_position = output_text.rfind(prompt[:-10:]) + 10
        return output_text[response_position:]

    
    def _tokenize_prompt(self, prompt: str) -> List:
        prompt_ids = self._tokenizer(prompt).input_ids
        if len(prompt_ids) > self.config.CONTEXT_SIZE:
            prompt_ids = prompt_ids[-self.config.CONTEXT_SIZE:]
        return prompt_ids
    
        