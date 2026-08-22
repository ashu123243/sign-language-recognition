from sign_language_detection.entity.config_entity import(
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

from sign_language_detection.constant.constants import (
    DATA_DIR,
    ARTIFACTS_DIR,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    TRAIN_VIDEO_DIR,
    VAL_VIDEO_DIR,
    TEST_VIDEO_DIR,
    DEV_SAMPLES_PER_CLASS,
    DEV_VAL_SAMPLES_PER_CLASS,
    NUM_CLASSES,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_FRAMES,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    TRANSFORMED_DATA_DIR,
    TRAIN_PROCESSED_CSV,
    VAL_PROCESSED_CSV,
    TEST_PROCESSED_CSV,
)


class ConfigurationManager:

    def __init__(self):
        self.data_dir=DATA_DIR
        self.artifacts_dir=ARTIFACTS_DIR

    def get_data_ingestion_config(self):
        config=DataIngestionConfig(
            data_dir=self.data_dir,
            artifacts_dir=self.artifacts_dir,
            train_csv=TRAIN_CSV,
            val_csv=VAL_CSV,
            test_csv=TEST_CSV,
            train_processed_csv=TRAIN_PROCESSED_CSV,
            val_processed_csv=VAL_PROCESSED_CSV,
            test_processed_csv=TEST_PROCESSED_CSV,
            train_video_dir=TRAIN_VIDEO_DIR,
            val_video_dir=VAL_VIDEO_DIR,
            test_video_dir=TEST_VIDEO_DIR,
            num_classes=NUM_CLASSES,
            dev_samples_per_class=DEV_SAMPLES_PER_CLASS,
            dev_val_samples_per_class=DEV_VAL_SAMPLES_PER_CLASS,
        )
        return config

    def get_data_validation_config(self):
        config=DataValidationConfig(
            train_processed_csv=TRAIN_PROCESSED_CSV,
            val_processed_csv=VAL_PROCESSED_CSV,
            test_processed_csv=TEST_PROCESSED_CSV,
            num_classes=NUM_CLASSES,
        )
        return config

    def get_data_transformation_config(self):
        config=DataTransformationConfig(
            train_processed_csv=TRAIN_PROCESSED_CSV,
            val_processed_csv=VAL_PROCESSED_CSV,
            test_processed_csv=TEST_PROCESSED_CSV,
            batch_size=BATCH_SIZE,
            num_frames = NUM_FRAMES,
            image_height = IMAGE_HEIGHT,
            image_width = IMAGE_WIDTH,
            transformed_data_dir=TRANSFORMED_DATA_DIR,
            train_video_dir=TRAIN_VIDEO_DIR,
            val_video_dir=VAL_VIDEO_DIR,
            test_video_dir=TEST_VIDEO_DIR,
        )
        return config

    def get_model_trainer_config(self):
        config=ModelTrainerConfig(
            num_classes=NUM_CLASSES,
            image_height = IMAGE_HEIGHT,
            image_width = IMAGE_WIDTH,
            num_frames=NUM_FRAMES,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            artifacts_dir=self.artifacts_dir
        )
        return config
    


        