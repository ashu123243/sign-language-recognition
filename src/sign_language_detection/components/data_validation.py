from sign_language_detection.entity.config_entity import DataValidationConfig
from sign_language_detection.exception.exception import CustomException
from sign_language_detection.logging.logger import logger

import sys
import pandas as pd

class DataValidation:
    
    def __init__(self,config):
        self.config=config
        
    def _load_prepared_data(self):
        
        try:
            train_processed_csv=self.config.train_processed_csv
            if not train_processed_csv.exists():
                raise Exception("Train CSV does not exist")
            
            val_processed_csv=self.config.val_processed_csv
            if not val_processed_csv.exists():
                raise Exception("Val CSV does not exist")
            
            test_processed_csv=self.config.test_processed_csv
            if not test_processed_csv.exists():
                raise Exception("Test CSV does not exist")
            
            logger.info("All CSV files are available")
            train_data=pd.read_csv(train_processed_csv)
            val_data=pd.read_csv(val_processed_csv)
            test_data=pd.read_csv(test_processed_csv)
            logger.info("All CSV files are loaded in DataFrames")
            return train_data,val_data,test_data
        except Exception as e:
            logger.error("Failed to load CSV files into DataFrames")
            raise CustomException(str(e),sys.exc_info())
    
    def data_sanity(self):
        
        try:
            expected_num_classes=self.config.num_classes
            expected_columns = ["filename", "label"]
            expected_labels = set(range(expected_num_classes))
            
            train_data,val_data,test_data=self._load_prepared_data()
            
            train_data_rows=train_data.shape[0]
            if train_data_rows==0:
                raise Exception("Training data is empty")
            if list(train_data.columns)!=expected_columns:
                raise Exception("Training data columns are invalid")
            if train_data["filename"].isna().any():
                raise Exception("Train data (filename) has missing values")
            if train_data["label"].isna().any():
                raise Exception("Train data (label) has missing labels")
            if train_data["filename"].duplicated().any():
                raise Exception("Train data contains duplicate filenames")
            train_data_unique_classes=train_data["label"].nunique()
            if train_data_unique_classes != expected_num_classes:
                raise Exception("Training data classes mismatch")
            
            val_data_rows=val_data.shape[0]
            if val_data_rows==0:
                raise Exception("Validation data is empty")
            if list(val_data.columns)!=expected_columns:
                raise Exception("Validation data columns are invalid")
            if val_data["filename"].isna().any():
                raise Exception("Val data (filename) has missing values")
            if val_data["label"].isna().any():
                raise Exception("Val data (label) has missing labels")
            if val_data["filename"].duplicated().any():
                raise Exception("Val data contains duplicate filenames")
            val_data_unique_classes=val_data["label"].nunique()
            if val_data_unique_classes != expected_num_classes:
                raise Exception("Validation data classes mismatch")
                                            
            test_data_rows=test_data.shape[0]
            if test_data_rows==0:
                raise Exception("Test data is empty")
            if list(test_data.columns)!=expected_columns:
                raise Exception("Test data columns are invalid")
            if test_data["filename"].isna().any():
                raise Exception("Test data (filename) has missing values")
            if test_data["label"].isna().any():
                raise Exception("Test data (label) has missing labels")
            if test_data["filename"].duplicated().any():
                raise Exception("Test data contains duplicate filenames")
            test_data_unique_classes=test_data["label"].nunique()
            if test_data_unique_classes != expected_num_classes:
                raise Exception("Test data classes mismatch")
            
            train_classes = set(train_data["label"])
            val_classes = set(val_data["label"])
            test_classes = set(test_data["label"])
            if train_classes!=val_classes:
                raise Exception("Training and Validation classes are not same")
            if val_classes!=test_classes:
                raise Exception("Testing and Validation classes are not same")
            if train_classes!=test_classes:
                raise Exception("Training and Testing classes are not same")
            if train_classes!=expected_labels:
                raise Exception("Training labels and expected labels are not same")
            if val_classes!=expected_labels:
                raise Exception("Validation labels and expected labels are not same")
            if test_classes!=expected_labels:
                raise Exception("Testing labels and expected labels classes are not same")
            
            
            
            train_files = set(train_data["filename"])
            val_files = set(val_data["filename"])
            test_files = set(test_data["filename"])
            if train_files.intersection(val_files):
                raise Exception("Data leakage between train and validation")
            if train_files.intersection(test_files):
                raise Exception("Data leakage between train and test")
            if val_files.intersection(test_files):
                raise Exception("Data leakage between test and validation")
            
            
            if not train_data["filename"].apply(lambda x: isinstance(x, str)).all():
                raise Exception("Train filenames must be strings")
            if train_data["filename"].str.strip().ne(train_data["filename"]).any():
                raise Exception("Train filenames contain leading/trailing whitespace")
            if not train_data["filename"].str.lower().str.endswith(".mp4").all():
                raise Exception("Train data contains invalid video file extension")
                
                        
            if not val_data["filename"].apply(lambda x: isinstance(x, str)).all():
                raise Exception("Validation filenames must be strings")
            if val_data["filename"].str.strip().ne(val_data["filename"]).any():
                raise Exception("Validation filenames contain leading/trailing whitespace")
            if not val_data["filename"].str.lower().str.endswith(".mp4").all():
                raise Exception("Validation data contains invalid video file extension")
            
            
            if not test_data["filename"].apply(lambda x: isinstance(x, str)).all():
                raise Exception("Test filenames must be strings")
            if test_data["filename"].str.strip().ne(test_data["filename"]).any():
                raise Exception("Test filenames contain leading/trailing whitespace")
            if not test_data["filename"].str.lower().str.endswith(".mp4").all():
                raise Exception("Test data contains invalid video file extension")
            
            
            if not pd.api.types.is_integer_dtype(train_data["label"]):
                raise Exception("Train labels must be integer")
            if not pd.api.types.is_integer_dtype(val_data["label"]):
                raise Exception("Validation labels must be integer")
            if not pd.api.types.is_integer_dtype(test_data["label"]):
                raise Exception("Test labels must be integer")
            
            
            train_class_counts = train_data.groupby("label").size()
            val_class_counts = val_data.groupby("label").size()
            test_class_counts = test_data.groupby("label").size()
            
            for label in expected_labels:
                count=train_class_counts.get(label,0)
                if count<=0:
                    raise Exception(f"Class {label} does not exist in training dataset")
            for label in expected_labels:
                count=val_class_counts.get(label,0)
                if count<=0:
                    raise Exception(f"Class {label} does not exist in validation dataset")
            for label in expected_labels:
                count=test_class_counts.get(label,0)
                if count<=0:
                    raise Exception(f"Class {label} does not exist in testing dataset")
            
            logger.info("Data Sanity checks completed")
            return True
            
        except Exception as e:
            logger.error("Error in data sanity check of data validation")
            raise CustomException(str(e),sys.exc_info())
            
    def initiate_data_validation(self):
        try:
            data_sanity_result=self.data_sanity()
            logger.info("Data Validation completed successfully")
            return data_sanity_result
        except Exception as e:
            logger.error("Data Validation failed")
            raise CustomException(str(e),sys.exc_info())
               
            