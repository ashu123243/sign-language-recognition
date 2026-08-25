from sign_language_detection.entity.config_entity import ModelTrainerConfig
from sign_language_detection.logging.logger import logger
from sign_language_detection.exception.exception import CustomException

import sys
import numpy as np
import tensorflow as tf
class ModelTrainer:
    
    def __init__(self,config):
        self.config=config
    
    def _get_batch_files(self,split_name):
        
        try:
            
            transformed_data_dir=self.config.transformed_data_dir
            split_transformed_data_dir=transformed_data_dir/split_name
            
            if not split_transformed_data_dir.exists():
                raise Exception(f"{split_transformed_data_dir} does not exist")
            
            batch_files = list(split_transformed_data_dir.glob("*.npz"))
            
            if len(batch_files)==0:
                raise Exception(f"No npz file found in {split_transformed_data_dir}")
            
            batch_files = sorted(batch_files)
            
            logger.info(f"Found {len(batch_files)} batch files for {split_name}")
            
            return batch_files
        
        except Exception as e:
            
            logger.error(f"Failed to get batch files for {split_name}")
            raise CustomException (str(e),sys.exc_info())
    
    def _load_batch(self,batch_path):
        
        try:
            
            data=np.load(batch_path)
            
            features = data["features"]
            labels = data["labels"]
            
            if len(features)==0:
                raise Exception(f"Features are empty for {batch_path}")
            
            if len(labels)==0:
                raise Exception(f"Labels are empty for {batch_path}")
            
            if len(features) != len(labels):
                raise Exception(f"Number of features and labels do not match for {batch_path}")
            
            logger.info(f"Loaded batch successfully: {batch_path}")
            
            return features,labels
            
        except Exception as e:
            
            logger.error(f"Failed to load batch: {batch_path}")
            raise CustomException (str(e),sys.exc_info())

    def _create_dataset(self,split_name):
        
        batch_files=self._get_batch_files(split_name)
        for batch_path in batch_files:
            features,labels=self._load_batch(batch_path)
            yield features, labels
    
    def _create_tf_dataset(self,split_name):
        
        generator = lambda: self._create_dataset(split_name)
        output_signature=(
            tf.TensorSpec(
                shape=(
                    None,
                    self.config.num_frames,
                    self.config.image_height,
                    self.config.image_width,
                    3
                ),
                dtype=tf.float32
            ),
            tf.TensorSpec(
                shape=(None,),
                dtype=tf.int64
            )
        )
        tf_dataset = tf.data.Dataset.from_generator(generator=generator,output_signature=output_signature)
        tf_dataset = tf_dataset.unbatch()
        tf_dataset = tf_dataset.batch(
            self.config.batch_size,
            drop_remainder=False
        )
        if split_name == "train":
            tf_dataset = tf_dataset.shuffle(buffer_size = 10)
        tf_dataset = tf_dataset.prefetch(
            tf.data.AUTOTUNE
        )
        return tf_dataset
     
    def _build_model(self):
        input_shape = (
            self.config.num_frames,
            self.config.image_height,
            self.config.image_width,
            3
        )       
        
        inputs = tf.keras.Input(shape=input_shape)
        
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Rescaling(scale=2.0, offset=-1.0)
        )(inputs)
        
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(self.config.image_height, self.config.image_width, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        
        base_model.trainable = True
        
        for layer in base_model.layers[:100]:
            layer.trainable = False
        
        x = tf.keras.layers.TimeDistributed(base_model)(x)
        
        x = tf.keras.layers.GRU(
            128,
            return_sequences=False
        )(x)
        
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        outputs = tf.keras.layers.Dense(
            self.config.num_classes,
            activation="softmax"
        )(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model
     
    def _compile_model(self,model):
        optimizer=tf.keras.optimizers.Adam(
            learning_rate = self.config.learning_rate
        )
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy()
        ]
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        return model
    
    def _train_model(self):
        
        train_dataset = self._create_tf_dataset("train")
        val_dataset = self._create_tf_dataset("val")
        
        model = self._build_model()
        model = self._compile_model(model)
        
        checkpoint_dir = self.config.checkpoint_dir

        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
        
        model_checkpoint=tf.keras.callbacks.ModelCheckpoint(
            filepath=self.config.checkpoint_file,
            monitor="val_loss",
            save_best_only=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
        
        callbacks = [early_stopping, model_checkpoint, reduce_lr]
        
        model.fit( 
                  train_dataset,
                  validation_data=val_dataset,
                  epochs=self.config.epochs,
                  callbacks=callbacks
                  )
        
        return model
    
    def _save_model(self,model):
        
        model_dir =self.config.model_dir
        if not model_dir.exists():
            model_dir.mkdir(parents=True, exist_ok=True)
        model_file = self.config.model_file  
        model.save(model_file)
        logger.info(f"Model saved successfully: {model_file}")
        return model_file
        
    def initiate_model_training(self):
        try:
            trained_model=self._train_model()
            model_file=self._save_model(trained_model)
            logger.info("Model training completed successfully")
            return model_file
        except Exception as e:
            logger.error("Model training failed")
            raise CustomException(str(e),sys.exc_info())
            
        
  