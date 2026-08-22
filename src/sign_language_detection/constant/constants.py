from pathlib import Path

# ==============================
# Project Root
# ==============================

PROJECT_ROOT=Path(__file__).resolve().parents[3]

# ==============================
# Data Directories
# ==============================

DATA_DIR=PROJECT_ROOT / "data"

RAW_DATA_DIR=DATA_DIR / "raw"
PROCESSED_DATA_DIR=DATA_DIR/"processed"
SPLITS_DIR=DATA_DIR/"splits"

# ==============================
# AUTSL Dataset
# ==============================

AUTSL_DIR=RAW_DATA_DIR/"AUTSL"

TRAIN_CSV = RAW_DATA_DIR / "train.csv"
VAL_CSV = RAW_DATA_DIR / "val.csv"
TEST_CSV = RAW_DATA_DIR / "test.csv"

TRAIN_VIDEO_DIR = AUTSL_DIR / "train"
VAL_VIDEO_DIR = AUTSL_DIR / "val"
TEST_VIDEO_DIR = AUTSL_DIR / "test"

# ==============================
# Artifacts & Logs
# ==============================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGS_DIR = PROJECT_ROOT / "logs"
TRANSFORMED_DATA_DIR=ARTIFACTS_DIR/"data_transformed"
TRAIN_PROCESSED_CSV = ARTIFACTS_DIR / "data" / "train_processed.csv"
VAL_PROCESSED_CSV = ARTIFACTS_DIR / "data" / "val_processed.csv"
TEST_PROCESSED_CSV = ARTIFACTS_DIR / "data" / "test_processed.csv"
# ==============================
# Dataset Configuration
# ==============================

NUM_CLASSES = 226

IMAGE_HEIGHT=224
IMAGE_WIDTH=224

NUM_FRAMES = 16

# ==============================
# Development Configuration
# ==============================

DEV_SAMPLES_PER_CLASS = 10
DEV_VAL_SAMPLES_PER_CLASS = 2

# ==============================
# Model Configuration
# ==============================

BATCH_SIZE = 16

EPOCHS = 20

LEARNING_RATE = 1e-4

# ==============================
# File Extensions
# ==============================

VIDEO_EXTENSION = ".mp4"
