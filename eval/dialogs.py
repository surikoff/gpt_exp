import torch
import config
from transformers import AutoTokenizer


REQUEST_TAG = "Вопрос: "
RESPONSE_TAG = "\nОтвет: "
STOP_WORDS = ["\n"]

class Interaction:
    def __init__(self, request: str = None, response: str = None):
        self._request = request
        self._response = response

    @property
    def request(self):
        return self._request if self._request is not None else ""
    
    @property
    def response(self):
        return self._response if self._response is not None else ""
    
    def __str__(self):
        return (REQUEST_TAG + self.request + RESPONSE_TAG + self.response).strip()



class PromptGenerator:
    def __init__(self, tokenizer: AutoTokenizer):
        self._tokenizer = tokenizer
        self._history = []

    @property
    def prompt(self):        
        if self._current_request is None:
            raise Exception("The request field must been set to generate prompt")
        dialog = self.dialog_history
        prompt = dialog + str(Interaction(self._current_request))
        return prompt.strip()
    
    @property
    def prompt_ids(self):                
        prompt_ids = self._tokenizer(self.prompt).input_ids
        if len(prompt_ids) > config.CONTEXT_SIZE:
            prompt_ids = prompt_ids[-config.CONTEXT_SIZE:]
        return prompt_ids

    @property
    def dialog_history(self):
        dialog_text = ""
        for interaction in self._history:
            if len(dialog_text) > 0:
                 dialog_text += "\n"
            dialog_text += str(interaction)
        return dialog_text


    def request(self, text: str):
        self._current_request = text


    def get_response(self, output: list[int]):
        output_text = self._tokenizer.decode(output)        
        response_position = output_text.rfind(RESPONSE_TAG) + len(RESPONSE_TAG)
        return output_text[response_position:]