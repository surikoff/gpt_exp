import torch
from train.gpt_trainer import GptTrainer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

DEFAULT_BATCH_SIZE = 1
DEFAULT_TEST_PART = 0.2


class GptTrainerLora(GptTrainer):
    def _init_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path, 
            quantization_config =  BitsAndBytesConfig(**self.config.bnb_config), 
            torch_dtype = torch.float16, 
            # device_map = "auto"
        )
        model.config.use_cache=False
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(**self.config.lora_config))
        print_trainable_parameters(model)
        return model



def print_trainable_parameters(model: AutoModelForCausalLM):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
