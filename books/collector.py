from random import sample, seed
from .dataset import BooksDataset
from .storage import BooksStorage


class BooksCollector:
    def __init__(self, books_storage: BooksStorage, chunk_size: int):
        self._books_storage = books_storage
        self._chunk_size = chunk_size
        self._chunks_number = self._books_storage.length // self._chunk_size        

    @property
    def length(self):
        return self._chunks_number

    def train_test_split(self, test_part: float = 0.2, shuffle: bool = True):        
        indexes = [i for i in range(self._chunks_number)]
        if shuffle:
            seed(0)
            indexes = sample(indexes, k = self._chunks_number)
        split_position = int(self.length * test_part)
        test_dataset = BooksDataset(self._books_storage, self._chunk_size, indexes[:split_position])
        train_dataset = BooksDataset(self._books_storage, self._chunk_size, indexes[split_position:])
        return  train_dataset, test_dataset