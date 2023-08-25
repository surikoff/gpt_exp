import torch
from .storage import BooksStorage


class BooksDataset(torch.utils.data.Dataset):
    def __init__(self, books_storage: BooksStorage, chunk_size: int, indexes: list[int]):
        self._books_storage = books_storage
        self._chunk_size = chunk_size
        self._indexes = indexes

    def __getitem__(self, index):
        target_index = self._indexes[index]
        position = target_index * self._chunk_size
        input_ids = self._books_storage.get_chunk(position, self._chunk_size)
        label_ids = [-100 if t == self._books_storage.eos_token_id else t for t in input_ids]
        attention_mask = [0 if t == self._books_storage.eos_token_id else 1 for t in input_ids]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }

    def __len__(self):
        return len(self._indexes)