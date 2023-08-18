import torch
import sys
import os.path
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
import config
from books.storage import BooksStorage
from books.collector import BooksCollector


def main(data_file: str, dump_folder: str, num_train_epochs: int):
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    books_storage = BooksStorage(tokenizer, data_file)
    books_collector = BooksCollector(books_storage, config.CONTEXT_SIZE)
    train_dataset, test_dataset = books_collector.train_test_split(config.TEST_PART)
    
    print(f"len(train_dataset) samples in train, {len(test_dataset)} samples in test")
    model = AutoModelForCausalLM.from_pretrained(config.MODEL_PATH)

    training_args = TrainingArguments(
        output_dir = config.OUTPUT_FOLDER,
        num_train_epochs = num_train_epochs,
        evaluation_strategy = "epoch",
        per_device_train_batch_size = config.BATCH_SIZE,
        per_device_eval_batch_size  = config.BATCH_SIZE
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )
    trainer.train()

    tokenizer.save_pretrained(dump_folder)
    model.save_pretrained(dump_folder)
    print(f"Fine-tuned model saved on {dump_folder}")


if __name__ == '__main__':
    try:
        data_file = sys.argv[1]
        if not os.path.isfile(data_file):
            raise Exception(f"Data file {data_file} not found or corrupted")
        dump_folder = sys.argv[2]
        num_train_epochs = int(sys.argv[3])
    except:
        raise Exception("Corrupted call, for example: python train.py temp/books.data finetuned_model/ 10")
    main(data_file, dump_folder, num_train_epochs)
