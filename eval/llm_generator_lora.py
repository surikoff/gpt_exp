from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftConfig, PeftModel
from eval.llm_generator import LLMGenerator


class LLMGeneratorLora(LLMGenerator):
    def __init__(self, model_dir: str, **kwargs):
        super().__init__(model_dir, **kwargs)

    def _init_model(self, model_dir: str) -> AutoModelForCausalLM:
        config = PeftConfig.from_pretrained(model_dir)           
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, 
            quantization_config =  BitsAndBytesConfig(**self.config.bnb_config), 
            torch_dtype = self.config.data_type,
            device_map = "auto"
        )
        model = PeftModel.from_pretrained(model, model_dir)
        return model