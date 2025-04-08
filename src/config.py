# Project Dir
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
CONVERTED_DATA_DIR = PROCESSED_DATA_DIR / "converted"

RAW_DATA_DIR = DATA_DIR / "raw"

SPLITS_DATA_DIR = DATA_DIR / "splits"
SPLIT_TRAIN_DIR = SPLITS_DATA_DIR / "train"
SPLIT_TEST_DIR = SPLITS_DATA_DIR / "test"
SPLIT_VAL_DIR = SPLITS_DATA_DIR / "val"

