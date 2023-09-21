import numpy as np
import torch
import os


"""
 Base GPT-type model, tested on: 
 - rugpt3 {small/medium/large}
 - ruGPT-3.5-13B
 - mGPT 1.3b/13b
"""

SOURCE_MODELS_FOLDER = "/mnt/ssd/models"
# SOURCE_MODELS_FOLDER = "/mnt/ram"
# SOURCE_MODELS_FOLDER = "../../models"
BASE_MODEL_NAME = "rugpt3small_based_on_gpt2"
# BASE_MODEL_NAME = "ruGPT-3.5-13B"
# BASE_MODEL_NAME = "ruGPT-3.5-13B-fp16"
# BASE_MODEL_NAME = "Llama-2-7b-chat-fp16"

BASE_MODEL_DIR= os.path.join(SOURCE_MODELS_FOLDER, BASE_MODEL_NAME)
BASE_MODEL_DTYPE = "auto"
WANDB_PROJECT = "gpt_finetune" 
WANDB_KEY="03d3a4d1d81c9713897283bc9bea0190253b1fa3"


# Base parameters for LLM model training
TRAIN_IN_FP16 = True
CONTEXT_SIZE = 1024 
TOKENS_DATATYPE = np.int32
TOKENS_DATASIZE = 4
MIN_DOC_LENGTH = 3000
TRAIN_BATCH_SIZE = 5
EVAL_BATCH_SIZE = 16
TEST_PART = 0.25
LEARNING_RATE = 1e-3  # 5e-4 for Lora and 5e-5 for ft

# Parameters for the LoRA training module
LORA_R = 32
LORA_ALPHA = 1
LORA_DROPOUT = 0.05

# Parameters for the BitSandBytes quantization module
BNB_LOAD_IN_4BIT = True
BNB_LOAD_IN_8BIT = False
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
    "train_in_fp16": TRAIN_IN_FP16,
    "learning_rate": LEARNING_RATE,
    "train_batch_size": TRAIN_BATCH_SIZE,
    "eval_batch_size": EVAL_BATCH_SIZE,
    "context_size": CONTEXT_SIZE,
    "test_part": TEST_PART,
}
    
BNB_CONFIG = {
    "load_in_4bit": BNB_LOAD_IN_4BIT,
    "load_in_8bit": BNB_LOAD_IN_8BIT,
    "bnb_4bit_quant_type": BNB_4BIT_QUANT_TYPE,
    "bnb_4bit_use_double_quant": BNB_4BIT_USE_DOUBLE_QUANT,
    "bnb_4bit_compute_dtype": BNB_4BIT_COMPUTE_DTYPE
}

LORA_CONFIG = {
    "r": LORA_R, 
    "lora_alpha": LORA_ALPHA,
    "lora_dropout": LORA_DROPOUT
}

EVAL_CONFIG = {
    "context_size": CONTEXT_SIZE,
    "stop_words": STOP_WORDS,
    "max_tokens": MAX_TOKENS,    
}

