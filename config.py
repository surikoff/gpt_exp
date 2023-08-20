import numpy as np
import torch


# MODEL_PATH = "/users/surikov/models/rugpt3small_based_on_gpt2"
MODEL_PATH = "/mnt/ssd/models/rugpt3small_based_on_gpt2"
# MODEL_PATH = "/mnt/ssd/models/ruGPT-3.5-13B-fp16"
OUTPUT_FOLDER = "temp"


# Base parameters for LLM model training
CONTEXT_SIZE = 2048
TOKENS_DATATYPE = np.int32
TOKENS_DATASIZE = 4
MIN_DOC_LENGTH = 100
BATCH_SIZE = 2
TEST_PART = 0.2

# Parameters for the LoRA training module
LORA_R = 8
LORA_ALPHA = 8
LORA_DROPOUT = 0.05

# Parameters for the BitSandBytes quantization module
BNB_LOAD_IN_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_USE_DOUBLE_QUANT = True
BNB_4BIT_COMPUTE_DTYPE = torch.bfloat16


TRAINING_CONFIG = {
    "model_path": MODEL_PATH,
    "batch_size": BATCH_SIZE,
    "output_folder": OUTPUT_FOLDER,
    "context_size": CONTEXT_SIZE,
    "test_part": TEST_PART,
}
    
BNB_CONFIG = {
    "load_in_4bit": BNB_LOAD_IN_4BIT,
    "bnb_4bit_quant_type": BNB_4BIT_QUANT_TYPE,
    "bnb_4bit_use_double_quant": BNB_4BIT_USE_DOUBLE_QUANT,
    "bnb_4bit_compute_dtype": BNB_4BIT_COMPUTE_DTYPE
}

LORA_CONFIG = {
    "r": LORA_R, 
    "lora_alpha": LORA_ALPHA,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

