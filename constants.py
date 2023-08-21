from argparse import Namespace


_TRAIN_MODES = {
    "NATIVE": "native",
    "LORA": "lora"
}
TRAIN_MODE = Namespace(**_TRAIN_MODES)

_EVAL_MODES = {
    "NATIVE": "native",
    "BNB": "bnb",
    "LORA": "lora"
}
EVAL_MODE = Namespace(**_EVAL_MODES)

