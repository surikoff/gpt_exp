import torch
import sys
import os.path
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import config
from books.storage import BooksStorage
from books.collector import BooksCollector


DEFAULT_BATCH_SIZE = 1
DEFAULT_TEST_PART = 0.2

class GptTrainer:
    def __init__(self, config: dict = {}):
        self._model_path = config["model_path"]
        self._batch_size = config["batch_size"] if "batch_size" in config else DEFAULT_BATCH_SIZE
        self._output_folder = config["output_folder"]
        self._data_file = config["data_file"]
        self._context_size = config["context_size"]
        self._test_part = config["test_part"] if "batch_size" in config else DEFAULT_TEST_PART
        self._mode = config["mode"] if "mode" in config else None
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        self._books_storage = BooksStorage(self._tokenizer, self._data_file)
        self._books_collector = BooksCollector(self._books_storage, self._context_size)
        self._train_dataset, self._test_dataset = self._books_collector.train_test_split(self._test_part)        
        print(f"{len(self._train_dataset)} samples in train, {len(self._test_dataset)} samples in test")

        if self._mode == "lora":
            self._model = self._init_lora_model()
        else:
            self._model = self._init_model()        
            
        

    def _init_model(self):
        model = AutoModelForCausalLM.from_pretrained(self._model_path)
        model.config.use_cache=False
        return model


    def _init_lora_model(self):
        nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
                )
        
        model = AutoModelForCausalLM.from_pretrained(self._model_path, quantization_config=nf4_config, torch_dtype=torch.float16, device_map="auto")
        model.config.use_cache=False
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        self.lora_config = LoraConfig(
            r = 8, 
            lora_alpha = 32, 
            # target_modules=["query_key_value"], 
            lora_dropout = 0.05, 
            bias = "none", 
            task_type = "CAUSAL_LM"
        )

        model = get_peft_model(model, self.lora_config)
        print_trainable_parameters(model)
        return model


    def train(self, num_train_epochs: int):
        training_args = TrainingArguments(
            output_dir = self._output_folder,
            num_train_epochs = num_train_epochs,
            evaluation_strategy = "epoch",
            per_device_train_batch_size = self._batch_size,
            per_device_eval_batch_size  = self._batch_size
        )
        trainer = Trainer(
            model = self._model,
            args = training_args,
            train_dataset = self._train_dataset,
            eval_dataset = self._test_dataset
        )
        return trainer.train()


    def save_model(self, dump_folder: str):
        self._tokenizer.save_pretrained(dump_folder)
        self._model.save_pretrained(dump_folder)
        print(f"Fine-tuned model saved on {dump_folder}")


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
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
