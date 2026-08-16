from pathlib import Path

# ==============================
# Project Root
# ==============================

PROJECT_ROOT=Path(__file__).resolve().parent[3]

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

# ==============================
# Dataset Configuration
# ==============================

NUM_CLASSES = 226

IMAGE_SIZE = (160, 160)

SEQUENCE_LENGTH = 16

# ==============================
# Development Configuration
# ==============================

DEV_SAMPLES_PER_CLASS = 10
DEV_VAL_SAMPLES_PER_CLASS = 2

# ==============================
# Model Configuration
# ==============================

BATCH_SIZE = 8

EPOCHS = 20

LEARNING_RATE = 1e-4

# ==============================
# File Extensions
# ==============================

VIDEO_EXTENSION = ".mp4"
