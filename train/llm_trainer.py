import os
import json
import torch
from argparse import Namespace
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from books.storage import BooksStorage
from books.collector import BooksCollector
from .utils import print_train_log, PrinterCallback


class LLMTrainer:
    def __init__(self, data_file: str, dump_folder: str, **kwargs):
        self._data_file = data_file
        self._dump_folder = dump_folder
        if not os.path.exists(self._dump_folder):
            os.mkdir(self._dump_folder)
        self.config = Namespace(**kwargs)        
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._books_storage = BooksStorage(self._tokenizer, self._data_file)
        self._books_collector = BooksCollector(self._books_storage, self.config.context_size)
        self._train_dataset, self._test_dataset = self._books_collector.train_test_split(self.config.test_part)        
        print(f"{len(self._train_dataset)} samples in train, {len(self._test_dataset)} samples in test")
        self._model = self._init_model()
        print(f"Model initialized in {self._model.device} with {self._model.dtype} weights")
        self._log_file = os.path.join(self._dump_folder, "train_log.json")
        self._printer = PrinterCallback(self._log_file)
        self._training_args = {
            "output_dir": os.path.join(self._dump_folder, "checkpoints"),
            "overwrite_output_dir": True,
            "learning_rate": self.config.learning_rate,
            "optim": "adafactor",
            "logging_steps": 1,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": True,
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            # "per_device_train_batch_size": self.config.train_batch_size,
            # "per_device_eval_batch_size": çself.config.eval_batch_size,
            "auto_find_batch_size": True,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "save_total_limit": 10,
            "fp16": self.config.train_in_fp16,
            "disable_tqdm": True
        }
        

    def _init_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            device_map = "auto",
            torch_dtype = self.config.model_dtype
        )
        model.resize_token_embeddings(len(self._tokenizer))
        model.config.use_cache=False
        return model

    def train(self, num_train_epochs: int):
        args = dict(self._training_args)
        args.update({"num_train_epochs": num_train_epochs})        
        trainer = Trainer(
            model = self._model,
            args = TrainingArguments(**args),
            train_dataset = self._train_dataset,
            eval_dataset = self._test_dataset,
            callbacks = [self._printer]
        )
        # print("Pretraining evaluation:", trainer.evaluate())
        trainer.train()
        print_train_log(self._log_file, os.path.join(self._dump_folder, "train_log.png"))
        


    def save_model(self):
        self._tokenizer.save_pretrained(self._dump_folder)
        self._model.save_pretrained(self._dump_folder)
        print(f"Fine-tuned model saved on {self._dump_folder}")


            
