from transformers import AutoTokenizer
import sys
import config
from books.storage import BooksStorage


def main(bools_folder: str, encoding: str, data_file: str):
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    books_storage = BooksStorage(
        tokenizer, 
        data_file, 
        erase_output_file = True)
    books_storage.from_txt_files(
        books_folder, 
        encoding = encoding)


if __name__ == '__main__':
    try:
        books_folder = sys.argv[1]
        encoding = sys.argv[2]
        data_file = sys.argv[3]
    except:
        raise Exception("Corrupted call, for example: python prepare_books.py books/ utf-8 temp/books.data")
    main(books_folder, encoding, data_file)