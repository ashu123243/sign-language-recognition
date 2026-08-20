from dataclasses import dataclass
# basically dataclass stored data/configuration in structured object
#in dataclass python automatically generate constructor
from pathlib import Path

from sign_language_detection.constant.constants import (
    DATA_DIR,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    TRAIN_VIDEO_DIR,
    VAL_VIDEO_DIR,
    TEST_VIDEO_DIR,
    DEV_SAMPLES_PER_CLASS,
    DEV_VAL_SAMPLES_PER_CLASS,
    NUM_CLASSES,
    IMAGE_SIZE,
    SEQUENCE_LENGTH,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    ARTIFACTS_DIR,
    )


@dataclass
class DataIngestionConfig:
    data_dir: Path
    artifacts_dir: Path
    
    train_csv: Path
    val_csv: Path
    test_csv: Path
    
    train_video_dir: Path
    val_video_dir: Path
    test_video_dir: Path
    
    num_classes: int
    dev_samples_per_class: int
    dev_val_samples_per_class: int


@dataclass
class DataValidationConfig:
    train_csv: Path
    val_csv: Path
    test_csv: Path
    num_classes: int


@dataclass
class DataTransformationConfig:
    image_size: tuple
    sequence_length: int
    batch_size: int


@dataclass
class ModelTrainerConfig:
    num_classes: int
    image_size: tuple
    sequence_length: int
    batch_size: int
    epochs: int
    learning_rate: float
    artifacts_dir: Path