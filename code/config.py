import os

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
LOAD_IN_4BIT = False
RANK = 16
ALPHA = 32
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
LR = 2e-4
MAX_SEQ_LEN = 4096
SEED = 42

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
