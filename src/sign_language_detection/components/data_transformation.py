from sign_language_detection.entity.config_entity import DataTransformationConfig
from sign_language_detection.logging.logger import logger
from sign_language_detection.exception.exception import CustomException

import sys
import pandas as pd
import cv2
import numpy as np

class DataTransformation:
    
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
    
    
    def _extract_frames(self,video_path):
        try:
            
            cap=cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise Exception(f"Unable to open video: {video_path}")
            
            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )
            
            if total_frames <= 0:
                cap.release()
                raise Exception(f"Video contains no frames: {video_path}")
            
            num_frames = self.config.num_frames
            if num_frames <= 0:
                raise Exception("Number of frames must be greater than zero")
            frame_indices = np.linspace(0,total_frames - 1,num_frames,dtype=int )
            
            frames = []
            
            for frame_index in frame_indices:

                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    frame_index
                )

                success, frame = cap.read()

                if not success:
                    cap.release()
                    raise Exception(
                        f"Failed to read frame {frame_index} "
                        f"from video: {video_path}"
                    )

                frames.append(frame)
            
            cap.release()

            return frames
        
        except Exception as e:
            logger.error(f"Failed to extract frames from video: {video_path}")
            raise CustomException(str(e),sys.exc_info())

    def _preprocess_frame(self,frame):
        
        try:
            target_height = self.config.image_height
            target_width = self.config.image_width

            if target_height <= 0 or target_width <= 0:
                raise Exception("Image height and width must be greater than zero")
            frame=cv2.resize(frame,(target_width,target_height))
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame=frame.astype(np.float32)/255.0
            return frame
        except Exception as e:
            logger.error("Failed to preprocess frame")
            raise CustomException(str(e), sys.exc_info())
        
    def _process_video(self,video_path):
        
        try:
            frames=self._extract_frames(video_path=video_path)
            processed_frames = []
            for frame in frames:
                preprocessed_frame=self._preprocess_frame(frame)
                processed_frames.append(preprocessed_frame)
            processed_frames=np.array(processed_frames)
            return processed_frames
        
        except Exception as e:
            logger.error(f"Failed to process video: {video_path}")
            raise CustomException(str(e),sys.exc_info())
    
    def _transform_dataset(self,data,video_dir,split_name):
        
        try:
            batch_size=self.config.batch_size
            if batch_size<=0:
                raise Exception("Batch size must be greater than zero")
            features=[]
            labels=[]
            batch_index = 0
            batch_saved=0
            skipped_videos = 0
            for _,row in data.iterrows():
                
                video_path=video_dir/row["filename"]
                try:
                    
                    if not video_path.exists():
                        raise Exception(f"Video does not exist: {video_path}")
                    processed_video=self._process_video(video_path)
                    features.append(processed_video)
                    labels.append(row["label"])
                except Exception as e:
                    skipped_videos+=1
                    logger.warning(
                        f"Skipping invalid video: {video_path} | "
                        f"Reason: {str(e)}"
                    )
                    continue
                if len(features) == batch_size:        
                    features = np.array(features)
                    labels = np.array(labels)
                    self._save_transformed_data(
                        features=features,
                        labels=labels,
                        split_name=split_name,
                        batch_index=batch_index
                    )
                    batch_index+=1
                    batch_saved+=1
                    features=[]
                    labels=[]
            if len(features)>0:
                features_array = np.array(features)
                labels_array = np.array(labels)

                self._save_transformed_data(
                    features=features_array,
                    labels=labels_array,
                    split_name=split_name,
                    batch_index=batch_index
                )
                batch_saved+=1
            logger.info(
                f"{split_name} transformation completed. "
                f"Batches saved: {batch_saved}, "
                f"Skipped videos: {skipped_videos}"
            )
                    

                
        except Exception as e:
            logger.error(f"Failed to transform dataset from: {video_dir}")
            raise CustomException(str(e),sys.exc_info())
        
    
    def _save_transformed_data(self,features,labels,split_name,batch_index):
        try:
            output_dir = self.config.transformed_data_dir / split_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"batch_{batch_index:03d}.npz"
            np.savez_compressed(
                output_file,
                features=features,
                labels=labels
            )

            logger.info(
                f"Saved {split_name} batch {batch_index}: {output_file}"
            )

        except Exception as e:
            logger.error(
                f"Failed to save {split_name} batch {batch_index}"
            )
            raise CustomException(str(e), sys.exc_info())
    
    def initiate_data_transformation(self):
        try:
            train_data,val_data,test_data=self._load_prepared_data()
            self._transform_dataset(train_data,self.config.train_video_dir,"train")
            self._transform_dataset(val_data,self.config.val_video_dir,"val")
            self._transform_dataset(test_data,self.config.test_video_dir,"test")
            logger.info("Data transformation completed successfully")
            return self.config.transformed_data_dir
        except Exception as e:
            logger.error("Data transformation failed")
            raise CustomException(str(e),sys.exc_info())