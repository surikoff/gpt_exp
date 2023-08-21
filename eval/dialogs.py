import torch
import config
from transformers import AutoTokenizer

REQUEST_TAG = "Вопрос: "
RESPONSE_TAG = "\nОтвет: "


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
    def __init__(self):        
        self._history = []
        self._current_request = None

    @property
    def prompt(self):        
        if self._current_request is None:
            raise Exception("The request method must been called before generate prompt")
        dialog = self.dialog_history
        prompt = dialog + "\n" + str(Interaction(self._current_request))
        return prompt.strip()    

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

    def response(self, text: str):
        if self._current_request is not None:
            self._history.append(Interaction(self._current_request, text))
            self._current_request = None
        else:
            raise Exception("Attempting to generate an interaction without specifying a request")

