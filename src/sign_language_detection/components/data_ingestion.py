'''               DATA INGESTION
                         │
                         ▼
              ConfigurationManager
                         │
                         ▼
               DataIngestionConfig
                         │
                         ▼
              ┌─────────────────────┐
              │ 1. Raw Data Check   │
              └─────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ 2. CSV Load         │
              └─────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ 3. Data Sanity      │
              │    Check            │
              └─────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ 4. Development      │
              │    Sampling         │
              └─────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ 5. Video Availability│
              │    Check             │
              └─────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ 6. Prepared Metadata│
              └─────────────────────┘
                         │
                         ▼
                    Artifacts/
                         │
                         ▼
                 Data Validation'''


from sign_language_detection.entity.config_entity import DataIngestionConfig
from sign_language_detection.logging.logger import logger
from sign_language_detection.exception.exception import CustomException

import sys
import pandas as pd

class DataIngestion:
    
    def __init__(self,config):
        self.config=config
        
        
    def helper_development_sampling(self,dataframe,samples_per_class):
        dataframe_groups=dataframe.groupby("label")
        selected_samples=[]
        for label,group in dataframe_groups:
            if group.shape[0]>=samples_per_class:
                selected_samples.append(group.sample(n=samples_per_class,random_state=42))
            else:
                size=group.shape[0]
                selected_samples.append(group.sample(n=size,random_state=42))
        return pd.concat(selected_samples, ignore_index=True)


    def _check_video_availability(self, dataframe, video_dir):
        available_samples=[]
        missing_files=[]
        for _,row in dataframe.iterrows():
            filename=row["filename"]
            video_path=video_dir/filename
            if video_path.exists():
                available_samples.append(row)
            else:
                missing_files.append(row)
                logger.warning(f"Video path: {video_path} not found")
                
        available_data=pd.DataFrame(available_samples)
        
        return available_data,missing_files
    
    
    def _save_prepared_metadata(self, train_data, val_data, test_data):
        try:
            train_output = self.config.train_processed_csv
            val_output = self.config.val_processed_csv
            test_output = self.config.test_processed_csv
            train_output.parent.mkdir(parents=True, exist_ok=True)
            train_data.to_csv(train_output, index=False)
            val_data.to_csv(val_output, index=False)
            test_data.to_csv(test_output, index=False)
            logger.info("Prepared metadata saved successfully")
            return train_output, val_output, test_output
        except Exception as e:
            logger.error("Failed to save prepared metadata")
            raise CustomException(str(e),sys.exc_info())
    
        
    def _check_raw_data(self):
        try:
            # Raw Data Check
            train_csv=self.config.train_csv
            val_csv=self.config.val_csv
            test_csv=self.config.test_csv

            if not train_csv.exists():
                raise Exception("Train CSV not found")
            if not val_csv.exists():
                raise Exception("Val CSV not found")
            if not test_csv.exists():
                raise Exception("Test CSV not found")
            logger.info("All CSV files found")
            
            # Video Directories Check
            
            train_video_dir = self.config.train_video_dir
            val_video_dir = self.config.val_video_dir
            test_video_dir = self.config.test_video_dir

            if not train_video_dir.exists():
                raise Exception(f"Train video directory not found: {train_video_dir}")
            if not val_video_dir.exists():
                raise Exception(f"Validation video directory not found: {val_video_dir}")
            if not test_video_dir.exists():
                raise Exception(f"Test video directory not found: {test_video_dir}")

            logger.info("Video directories are ready")
            
        except Exception as e:
            logger.error("Raw data check failed")
            raise CustomException(str(e),sys.exc_info())  
    
             
    def prepare_data(self):
    
        try:
            # CSV Load
            train_csv=self.config.train_csv
            val_csv=self.config.val_csv
            test_csv=self.config.test_csv
            
            train_data = pd.read_csv(
                train_csv,
                header=None,
                names=["filename", "label"]
            )

            val_data = pd.read_csv(
                val_csv,
                header=None,
                names=["filename", "label"]
            )

            test_data = pd.read_csv(
                test_csv,
                header=None,
                names=["filename", "label"]
            )
            
            logger.info("CSV files loaded successfully")

        except Exception as e:
            logger.error("CSV files were not loaded successfully")
            raise CustomException(str(e),sys.exc_info())
    
        
        try:
            
            # Data Sanity Check
            expected_num_classes = self.config.num_classes
            expected_columns = ["filename", "label"]
            expected_labels = set(range(expected_num_classes))
            
            train_data_rows=train_data.shape[0]
            if train_data_rows==0:
                raise Exception("Training data is empty")
            train_data_unique_classes=train_data["label"].nunique()
            if train_data_unique_classes != expected_num_classes:
                raise Exception("Training data classes mismatch")
            if list(train_data.columns)!=expected_columns:
                raise Exception("training data columns are invalid")
            if train_data["filename"].isna().any():
                raise Exception("Train data (filename) has missing values")
            if train_data["filename"].astype(str).str.strip().eq("").any():
                raise Exception("Train data has empty filenames")
            if train_data["label"].isna().any():
                raise Exception("Train data (label) has missing labels")
            if train_data["filename"].duplicated().any():
                raise Exception("Train data contains duplicate filenames")
            train_labels = set(train_data["label"])
            if train_labels != expected_labels:
                raise Exception("Training data contains invalid labels")
            
            
            val_data_rows=val_data.shape[0]
            if val_data_rows==0:
                raise Exception("Validation data is empty")
            val_data_unique_classes=val_data["label"].nunique()
            if val_data_unique_classes != expected_num_classes:
                raise Exception("Validation data classes mismatch")
            if list(val_data.columns)!=expected_columns:
                raise Exception("validation data columns are invalid")
            if val_data["filename"].isna().any():
                raise Exception("Val data (filename) has missing values")
            if val_data["filename"].astype(str).str.strip().eq("").any():
                raise Exception("Val data has empty filenames")
            if val_data["label"].isna().any():
                raise Exception("Val data (label) has missing labels")
            if val_data["filename"].duplicated().any():
                raise Exception("Val data contains duplicate filenames")
            val_labels = set(val_data["label"])
            if val_labels != expected_labels:
                raise Exception("Validation data contains invalid labels")
                        
                        
            test_data_rows=test_data.shape[0]
            if test_data_rows==0:
                raise Exception("Test data is empty")
            test_data_unique_classes=test_data["label"].nunique()
            if test_data_unique_classes != expected_num_classes:
                raise Exception("Test data classes mismatch")
            if list(test_data.columns)!=expected_columns:
                raise Exception("testing data columns are invalid")
            if test_data["filename"].isna().any():
                raise Exception("Test data (filename) has missing values")
            if test_data["filename"].astype(str).str.strip().eq("").any():
                raise Exception("Test data has empty filenames")
            if test_data["label"].isna().any():
                raise Exception("Test data (label) has missing labels")
            if test_data["filename"].duplicated().any():
                raise Exception("Test data contains duplicate filenames")
            test_labels = set(test_data["label"])
            if test_labels != expected_labels:
                raise Exception("Testing data contains invalid labels")
                  
            
            train_files = set(train_data["filename"])
            val_files = set(val_data["filename"])
            test_files = set(test_data["filename"])
            if train_files.intersection(val_files):
                raise Exception("Data leakage between train and validation")
            if train_files.intersection(test_files):
                raise Exception("Data leakage between train and test")
            if val_files.intersection(test_files):
                raise Exception("Data leakage between test and validation")
            
            train_classes = set(train_data["label"])
            val_classes = set(val_data["label"])
            test_classes = set(test_data["label"])
            if train_classes!=val_classes:
                raise Exception("Training and Validation classes are not same")
            if val_classes!=test_classes:
                raise Exception("Testing and Validation classes are not same")
            if train_classes!=test_classes:
                raise Exception("Training and Testing classes are not same")
            
        except Exception as e:
            logger.error("Data sanity check failed")
            raise CustomException(str(e),sys.exc_info())
        
        try:
            # Development Sampling            
            dev_samples_per_class=self.config.dev_samples_per_class
            dev_train_data=self.helper_development_sampling(train_data,dev_samples_per_class)
            
            dev_val_samples_per_class=self.config.dev_val_samples_per_class
            dev_val_data=self.helper_development_sampling(val_data,dev_val_samples_per_class)
            
            logger.info(f"Development train data shape: {dev_train_data.shape}")
            logger.info(f"Development validation data shape: {dev_val_data.shape}")
                        
        except Exception as e:
            logger.error("Development sampling failed")
            raise CustomException(str(e),sys.exc_info())
        
        try:
            
            # Video Availability Check 
            available_train_data, missing_train_files = self._check_video_availability(
                dev_train_data,
                self.config.train_video_dir
            )
            if available_train_data.empty:
                raise Exception("No Training videos are available")
            available_train_videos=available_train_data.shape[0]
            logger.info(f"Available train videos : {available_train_videos}")
            logger.info(f"Missing train videos : {len(missing_train_files)}")
            
            
            available_val_data, missing_val_files = self._check_video_availability(
                dev_val_data,
                self.config.val_video_dir
            )
            if available_val_data.empty:
                raise Exception("No Validation videos are available")
            available_val_videos=available_val_data.shape[0]
            logger.info(f"Available val videos : {available_val_videos}")
            logger.info(f"Missing val videos : {len(missing_val_files)}")
            
            
            available_test_data, missing_test_files = self._check_video_availability(
                test_data,
                self.config.test_video_dir
            )
            if available_test_data.empty:
                raise Exception("No Testing videos are available")
            available_test_videos=available_test_data.shape[0]
            logger.info(f"Available test videos : {available_test_videos}")
            logger.info(f"Missing test videos : {len(missing_test_files)}")
            
            train_prepared_labels = set(available_train_data["label"])
            val_prepared_labels = set(available_val_data["label"])
            test_prepared_labels = set(available_test_data["label"])
            if train_prepared_labels != expected_labels:
                raise Exception("Training prepared data has missing or invalid classes")
            if val_prepared_labels != expected_labels:
                raise Exception("Validation prepared data has missing or invalid classes")
            if test_prepared_labels != expected_labels:
                raise Exception("Testing prepared data has missing or invalid classes")
            
            logger.info("Prepared data validation completed")
            
        except Exception as e:
            logger.error("error in Video Availability Check ")
            raise CustomException(str(e),sys.exc_info())
            
        train_output, val_output, test_output=self._save_prepared_metadata(
                                                available_train_data,
                                                available_val_data,
                                                available_test_data)
            
        return train_output, val_output, test_output
        
    

    def initiate_data_ingestion(self):
        try:
            self._check_raw_data()
            train_output, val_output, test_output=self.prepare_data()
            logger.info("Data Ingestion Completed")
            return train_output, val_output, test_output
        except Exception as e:
            logger.error("error in Data Ingestion")
            raise CustomException(str(e),sys.exc_info())
