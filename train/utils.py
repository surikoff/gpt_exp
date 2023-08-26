import torch
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TrainerCallback
import json
import plotly.graph_objects as go 


class PrinterCallback(TrainerCallback):
    def __init__(self, logfile: str):
        self._logfile = logfile

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_local_process_zero:            
            with open(self._logfile, "w") as f:
                f.write(json.dumps(state.log_history))

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True

                

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


def print_train_log(data_file: str, target_file : str):
    with open(data_file) as f:
        data = json.load(f)

    train_x = []
    train_y = []
    eval_x = []
    eval_y = []
    for record in data:
        if "loss" in record:
            train_x.append(record["epoch"])
            train_y.append(record["loss"])
        if "eval_loss" in record:
            eval_x.append(record["epoch"])
            eval_y.append(record["eval_loss"])


    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=train_x,
            y=train_y,
            name="Training Loss",
            mode="lines",
            line=dict(color="Blue"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=eval_x,
            y=eval_y,
            name="Validation Loss",
            mode="lines",
            line=dict(color="Red", width=3),
        )
    )

    fig.update_layout(
        title="Train log", xaxis_title="Epoch", yaxis_title="Loss"
    )

    fig.write_image(target_file)
    print("Train statistic saved in", target_file)
