from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')


PROJ_ROOT = Path(__file__).resolve().parents[0]
logging.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJ_ROOT / "models"
TRAIN_DIR = PROJ_ROOT / "train"
CHECKPOINTS_DIR = TRAIN_DIR / "checkpoints"

