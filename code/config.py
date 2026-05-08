import os

MODEL_NAME = "Qwen/Qwen2.5-3B"
TRUST_REMOTE_CODE = False
LOAD_IN_4BIT = False
RANK = 8
ALPHA = 32
DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]
LR = 1e-4
MAX_SEQ_LEN = 4096
SEED = 42
EPOCHS = 1
BATCH_SIZE = 16          # effective batch size (via grad accum)
PER_DEVICE_BATCH_SIZE = 1
WARMUP_RATIO = 0.03
SAVE_STRATEGY = "steps"
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 5

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
