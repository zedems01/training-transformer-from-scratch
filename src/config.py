from pathlib import Path
import logging

PROJ_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJ_ROOT / "src"
DATA_DIR = SRC_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = SRC_DIR / "models"
TRAIN_DIR = SRC_DIR / "train"
CHECKPOINTS_DIR = TRAIN_DIR / "checkpoints"

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
    logging.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

if __name__ == "__main__":
    main()

