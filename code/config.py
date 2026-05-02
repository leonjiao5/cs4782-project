import os

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
LOAD_IN_4BIT = False
RANK = 16
ALPHA = 32
DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]
LR = 2e-4
MAX_SEQ_LEN = 4096
SEED = 42
EPOCHS = 3
BATCH_SIZE = 16          # effective batch size (via grad accum)
PER_DEVICE_BATCH_SIZE = 1
WARMUP_RATIO = 0.03
SAVE_STRATEGY = "steps"
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 5

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
