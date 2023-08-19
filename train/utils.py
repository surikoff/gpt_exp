import torch
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids: list):
        self.keywords = keywords_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] in self.keywords:
            return True
        return False
    

def generate_stopping_criteria_list(tokenizer: AutoTokenizer, stop_words: list):
    stop_ids = [tokenizer.encode(word)[0] for word in stop_words]
    stop_criteria = KeywordsStoppingCriteria(stop_ids)
    return StoppingCriteriaList([stop_criteria])