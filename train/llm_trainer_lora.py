import torch
from train.llm_trainer import LLMTrainer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftConfig, PeftModel, TaskType
DEFAULT_BATCH_SIZE = 1
DEFAULT_TEST_PART = 0.2


class LLMTrainerLora(LLMTrainer):
    def _init_model(self):

        if self.config.lora_weights is not None:
            config = PeftConfig.from_pretrained(self.config.lora_weights)
            model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path, 
                quantization_config =  BitsAndBytesConfig(**self.config.bnb_config), 
                torch_dtype = self.config.model_dtype,
                device_map = "auto"
            )
            model.resize_token_embeddings(len(self._tokenizer))
            model = PeftModel.from_pretrained(model, self.config.lora_weights)            
            model.config.use_cache=False
            model.gradient_checkpointing_enable()
            model = prepare_model_for_kbit_training(model)
        else:    
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path, 
                quantization_config =  BitsAndBytesConfig(**self.config.bnb_config), 
                torch_dtype = self.config.model_dtype,
                device_map = "auto"
            )
            model.resize_token_embeddings(len(self._tokenizer))
            model.config.use_cache=False
            model.gradient_checkpointing_enable()
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, LoraConfig(
                task_type = TaskType.CAUSAL_LM,
                inference_mode = False, 
                **self.config.lora_config))
        
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
