import os
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
        self._books_storage = BooksStorage(self._tokenizer, self._data_file)
        self._books_collector = BooksCollector(self._books_storage, self.config.context_size)
        self._train_dataset, self._test_dataset = self._books_collector.train_test_split(self.config.test_part)        
        print(f"{len(self._train_dataset)} samples in train, {len(self._test_dataset)} samples in test")
        self._model = self._init_model()
        self._printer = PrinterCallback(os.path.join(self._dump_folder, "train.log"))
        self._training_args = TrainingArguments(
            output_dir = os.path.join(self._dump_folder, "checkpoints"),
            overwrite_output_dir = True,
            logging_steps = 1,
            num_train_epochs = DEFAULT_EPOCHS_NUMBER,
            gradient_accumulation_steps = 4,
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            per_device_train_batch_size = self.config.batch_size,
            per_device_eval_batch_size  = self.config.batch_size,
            load_best_model_at_end = True,
            metric_for_best_model = 'eval_loss',
            greater_is_better = False,
            save_total_limit = 10
        )
        

    def _init_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype = self.config.data_type
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
        return trainer.train()    


    def save_model(self):
        self._tokenizer.save_pretrained(self._dump_folder)
        self._model.save_pretrained(self._dump_folder)
        print(f"Fine-tuned model saved on {self._dump_folder}")


class PrinterCallback(TrainerCallback):
    def __init__(self, filepath):
        self._file_handle = open(filepath, 'w')

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            if 'epoch' in logs and 'loss' in logs:
                self._file_handle.write('{}\t{}\n'.format(logs['epoch'], logs['loss']))
                self._file_handle.flush()