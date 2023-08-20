from argparse import Namespace
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from books.storage import BooksStorage
from books.collector import BooksCollector


class GptTrainer:
    def __init__(self, data_file: str, **kwargs):
        self._data_file = data_file
        self.config = Namespace(**kwargs)        
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self._books_storage = BooksStorage(self._tokenizer, self._data_file)
        self._books_collector = BooksCollector(self._books_storage, self.config.context_size)
        self._train_dataset, self._test_dataset = self._books_collector.train_test_split(self.config.test_part)        
        print(f"{len(self._train_dataset)} samples in train, {len(self._test_dataset)} samples in test")
        self._model = self._init_model()
        
    def _init_model(self):
        model = AutoModelForCausalLM.from_pretrained(self.config.model_path)
        model.config.use_cache=False
        return model

    def train(self, num_train_epochs: int):
        training_args = TrainingArguments(
            output_dir = self.config.output_folder,
            num_train_epochs = num_train_epochs,
            gradient_accumulation_steps = 4,
            evaluation_strategy = "epoch",
            per_device_train_batch_size = self.config.batch_size,
            per_device_eval_batch_size  = self.config.batch_size
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
