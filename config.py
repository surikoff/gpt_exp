import numpy as np


MODEL_PATH = "/users/surikov/models/rugpt3small_based_on_gpt2"
# MODEL_PATH = "/mnt/ssd/models/rugpt3small_based_on_gpt2"
OUTPUT_FOLDER = "temp"
CONTEXT_SIZE = 2048
TOKENS_DATATYPE = np.int32
TOKENS_DATASIZE = 4
MIN_DOC_LENGTH = 100
BATCH_SIZE = 2
TEST_PART = 0.2