from argparse import Namespace


_TRAIN_MODES = {
    "NATIVE": "native",
    "BNB": "bnb",
    "LORA": "lora"
}
TRAIN_MODE = Namespace(**_TRAIN_MODES)

_DEVICES = {
    "CPU": "cpu",
    "GPU": "gpu"
}
DEVICE = Namespace(**_DEVICES)