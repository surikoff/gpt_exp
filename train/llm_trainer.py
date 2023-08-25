import os
import json
import torch
from argparse import Namespace
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer, TrainerCallback
from books.storage import BooksStorage
from books.collector import BooksCollector

DEFAULT_EPOCHS_NUMBER = 1


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
        self._printer = PrinterCallback(os.path.join(self._dump_folder, "train_log.json"))
        self._training_args = TrainingArguments(
            output_dir = os.path.join(self._dump_folder, "checkpoints"),
            overwrite_output_dir = True,
            learning_rate = self.config.learning_rate,
            # optim="adafactor",
            logging_steps = 1,
            num_train_epochs = DEFAULT_EPOCHS_NUMBER,
            # gradient_accumulation_steps = 1,
            # gradient_checkpointing = True,
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            per_device_train_batch_size = self.config.train_batch_size,
            per_device_eval_batch_size  = self.config.eval_batch_size,
            # auto_find_batch_size = True,
            load_best_model_at_end = True,
            metric_for_best_model = 'eval_loss',
            greater_is_better = False,
            save_total_limit = 10,
            # fp16 = True,
            disable_tqdm = True
        )
        

    def _init_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            device_map = "auto",
            torch_dtype = self.config.model_dtype
        )
        model.config.use_cache=False
        return model

    def train(self, num_train_epochs: int):
        self._training_args.num_train_epochs = num_train_epochs
        trainer = Trainer(
            model = self._model,
            args = self._training_args,
            train_dataset = self._train_dataset,
            eval_dataset = self._test_dataset,
            callbacks = [self._printer]
        )
        print("Pretraining evaluation:", trainer.evaluate())
        trainer.train()        


    def save_model(self):
        self._tokenizer.save_pretrained(self._dump_folder)
        self._model.save_pretrained(self._dump_folder)
        print(f"Fine-tuned model saved on {self._dump_folder}")


class PrinterCallback(TrainerCallback):
    def __init__(self, logfile: str):
        self._logfile = logfile

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_local_process_zero:            
            with open(self._logfile, "w") as f:
                f.write(json.dumps(state.log_history))
            
