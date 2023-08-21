from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from eval.generator import LLMGenerator


class LLMGeneratorBNB(LLMGenerator):
    def __init__(self, model_dir: str, **kwargs):
        super().__init__(model_dir, **kwargs)

    def _init_model(self, model_dir: str) -> AutoModelForCausalLM:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config =  BitsAndBytesConfig(**self.config.bnb_config), 
            torch_dtype = self.config.data_type,
            device_map = "auto"
        )
        return model