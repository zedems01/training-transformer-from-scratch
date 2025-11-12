from pathlib import Path
import logging

PROJ_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJ_ROOT / "src"
DATA_DIR = SRC_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SP_MODEL_PATH = PROCESSED_DATA_DIR / "sp_bpe.model"
TRAIN_SRC_FILE = PROCESSED_DATA_DIR / "train.en.bpe"
TRAIN_TGT_FILE = PROCESSED_DATA_DIR / "train.fr.bpe"
VALID_SRC_FILE = PROCESSED_DATA_DIR / "valid.en.bpe"
VALID_TGT_FILE = PROCESSED_DATA_DIR / "valid.fr.bpe"
TEST_SRC_FILE = PROCESSED_DATA_DIR / "test.en.bpe"
TEST_TGT_FILE = PROCESSED_DATA_DIR / "test.fr.bpe"
TEST_TGT_RAW_FILE = PROCESSED_DATA_DIR / "test.fr"
SCRIPTS_DIR = SRC_DIR / "scripts"
CHECKPOINTS_DIR = SCRIPTS_DIR / "checkpoints"

UNK_ID = 0
PAD_ID = 1
BOS_ID = 2
EOS_ID = 3

# D_MODEL = 512
# NUM_HEADS = 8
# D_FF = 2048
# NUM_LAYERS = 6
# DROPOUT = 0.1
# BATCH_SIZE = 16
# EVAL_BATCH_SIZE = 16
# EPOCHS = 10
# LR = 5e-4
# MAX_SEQ_LEN = 512
# WARMUP_STEPS = 4000
# EARLY_STOPPING_PATIENCE = 3
# NUM_WORKERS = 0

D_MODEL = 256
NUM_HEADS = 2
D_FF = 512
NUM_LAYERS = 1
DROPOUT = 0.1
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 16
EPOCHS = 1
LR = 5e-4
MAX_SEQ_LEN = 128
WARMUP_STEPS = 4000
EARLY_STOPPING_PATIENCE = 3
NUM_WORKERS = 0

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
    logging.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

if __name__ == "__main__":
    main()

