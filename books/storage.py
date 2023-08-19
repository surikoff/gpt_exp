import os
import numpy as np
import config
from transformers import AutoTokenizer


class BooksStorage:
    def __init__(self, tokenizer: AutoTokenizer, output_file: str, erase_output_file: bool = False):
        self.tokenizer = tokenizer
        self._output_file = output_file
        self._min_doc_len = config.MIN_DOC_LENGTH
        self._tokens_datatype = config.TOKENS_DATATYPE
        self._tokens_datasize = config.TOKENS_DATASIZE
        self._length = 0
        if os.path.isfile(self._output_file):
            if erase_output_file:
                os.remove(self._output_file)
            else:
                self._length = os.path.getsize(self._output_file) // self._tokens_datasize

    def from_txt_files(self, folder: str, encoding: str = "utf8"):
        num = 0
        print(f"Strarting to process folder '{folder}':")
        
        for file in os.listdir(folder):
            if file.endswith(".txt"):
                print(f"Process file '{file}'...")
                file_path = os.path.join(folder, file)
                text = self._read_txt_file(file_path, encoding)
                self._process_text(text)
                num += 1
        print(f"Finish: {num} files processed, {self.length} tokens collected.")
    
    def _read_txt_file(self, file_path: str, encoding: str):
        with open(file_path, "r", encoding=encoding) as f:
            return(f.read())
        
    def _process_text(self, text: str):
        docs = []
        for part in text.split("\n"):
            if len(part) >= self._min_doc_len:
                docs.append(part)
        for doc_ids in self.tokenizer(docs, padding=False).input_ids:
            self._write_to_output_file(doc_ids + [self.tokenizer.eos_token_id])

    def _write_to_output_file(self, tokens: list):
        part = np.array(tokens, dtype=self._tokens_datatype)
        with open(self._output_file, "ab") as f:
            part.tofile(f)
        self._length += len(part)
        

    def get_chunk(self, position: int, length: int) -> list[int]:
        return np.fromfile(
            self._output_file, 
            dtype = self._tokens_datatype, 
            offset = self._tokens_datasize * position,
            count = length)
        

    @property
    def length(self) -> int:
        return self._length


