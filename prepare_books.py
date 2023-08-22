import argparse
from transformers import AutoTokenizer
import config
from books.storage import BooksStorage


def main(books_folder: str, encoding: str, data_file: str):
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_DIR)
    books_storage = BooksStorage(
        tokenizer, 
        data_file, 
        erase_output_file = True)
    books_storage.from_txt_files(
        books_folder, 
        encoding = encoding)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to collect books for training LLM models")
    parser.add_argument('input', type=str, help='Source folder with books in txt format')
    parser.add_argument('output', type=str, help='Output file to save the collected index')
    parser.add_argument('--encoding', type=str, required=False, default="utf-8", help='Encoding format, default: utf-8')
    args = parser.parse_args()

    main(args.input, args.encoding, args.output)