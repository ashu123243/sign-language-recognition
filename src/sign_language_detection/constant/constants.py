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
if Path("/kaggle/input").exists():
    TRANSFORMED_DATA_DIR = Path(
        "/kaggle/input/datasets/ashutoshpal007/sign-language-transformed-data"
    )
else:
    TRANSFORMED_DATA_DIR = ARTIFACTS_DIR / "data_transformed"
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

BATCH_SIZE = 1
EPOCHS = 12

LAYER3_LEARNING_RATE = 5e-6
LAYER4_LEARNING_RATE = 1e-5
FC_LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
GRADIENT_CLIP_VALUE = 1.0

SCHEDULER_T0 = 10
SCHEDULER_T_MULT = 2
SCHEDULER_ETA_MIN = 1e-6

DROPOUT_RATE = 0.5

MODEL_DIR = ARTIFACTS_DIR / "model"

MODEL_FILE = MODEL_DIR / "sign_language_model.pth"

CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
CHECKPOINT_FILE = CHECKPOINT_DIR / "best_model.pth"

# ==============================
# File Extensions
# ==============================

VIDEO_EXTENSION = ".mp4"
