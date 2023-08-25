import numpy as np
import torch
from peft import LoraConfig, TaskType


# BASE_MODEL_DIR= "/users/surikov/models/rugpt3small_based_on_gpt2"
# BASE_MODEL_DIR= "/users/surikov/models/rugpt3medium_based_on_gpt2"
# BASE_MODEL_DIR= "/mnt/ssd/models/rugpt3small_based_on_gpt2"
BASE_MODEL_DIR= "/mnt/ssd/models/rugpt3medium_based_on_gpt2"
# BASE_MODEL_DIR= "/mnt/ssd/models/rugpt3large_based_on_gpt2"
# BASE_MODEL_DIR = "/mnt/ssd/models/ruGPT-3.5-13B-fp16"
BASE_MODEL_DTYPE = torch.float32


# Base parameters for LLM model training
CONTEXT_SIZE = 1024
TOKENS_DATATYPE = np.int32
TOKENS_DATASIZE = 4
MIN_DOC_LENGTH = 100
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
TEST_PART = 0.4
LEARNING_RATE = 1e-4

# Parameters for the LoRA training module
LORA_R = 32
LORA_ALPHA = 1
LORA_DROPOUT = 0.05

# Parameters for the BitSandBytes quantization module
BNB_LOAD_IN_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_USE_DOUBLE_QUANT = True
BNB_4BIT_COMPUTE_DTYPE = torch.bfloat16

# Parameters for evaluation
STOP_WORDS = [".", "!", "?", "\n"]
# STOP_WORDS = ["\n"]
MAX_TOKENS = 100


TRAINING_CONFIG = {
    "model_path": BASE_MODEL_DIR,
    "model_dtype": BASE_MODEL_DTYPE,
    "learning_rate": LEARNING_RATE,
    "train_batch_size": TRAIN_BATCH_SIZE,
    "eval_batch_size": EVAL_BATCH_SIZE,
    "context_size": CONTEXT_SIZE,
    "test_part": TEST_PART,
}
    
BNB_CONFIG = {
    "load_in_4bit": BNB_LOAD_IN_4BIT,
    "bnb_4bit_quant_type": BNB_4BIT_QUANT_TYPE,
    "bnb_4bit_use_double_quant": BNB_4BIT_USE_DOUBLE_QUANT,
    "bnb_4bit_compute_dtype": BNB_4BIT_COMPUTE_DTYPE
}

LORA_CONFIG = LoraConfig(
    task_type = TaskType.CAUSAL_LM, 
    inference_mode=False,
    r = LORA_R, 
    lora_alpha = LORA_ALPHA,
    lora_dropout = LORA_DROPOUT,
    # bias = "lora_only"
)

EVAL_CONFIG = {
    "context_size": CONTEXT_SIZE,
    "stop_words": STOP_WORDS,
    "max_tokens": MAX_TOKENS,    
}

