'''START

1. Import dataclass.

2. Import Path.

3. Import required constants from:
   sign_language_detection.constant.constants

4. Create a dataclass:
   DataIngestionConfig

5. Inside DataIngestionConfig define:

   a. source dataset identifier
      → AUTSL Kaggle dataset name

   b. train CSV path
      → TRAIN_CSV

   c. validation CSV path
      → VAL_CSV

   d. test CSV path
      → TEST_CSV

   e. train video directory
      → TRAIN_VIDEO_DIR

   f. validation video directory
      → VAL_VIDEO_DIR

   g. test video directory
      → TEST_VIDEO_DIR

   h. development samples per class
      → DEV_SAMPLES_PER_CLASS

   i. development validation samples per class
      → DEV_VAL_SAMPLES_PER_CLASS

6. Create another dataclass:
   DataValidationConfig

7. Give it configuration needed for validation:

   a. train CSV path
   b. validation CSV path
   c. test CSV path
   d. expected number of classes
      → NUM_CLASSES

8. Create another dataclass:
   DataTransformationConfig

9. Give it:

   a. image size
   b. sequence length
   c. batch size

10. Create another dataclass:
    ModelTrainerConfig

11. Give it:

    a. number of classes
    b. image size
    c. sequence length
    d. batch size
    e. epochs
    f. learning rate
    g. model output/artifact directory

END'''

