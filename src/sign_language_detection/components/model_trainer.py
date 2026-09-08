from sign_language_detection.entity.config_entity import ModelTrainerConfig
from sign_language_detection.logging.logger import logger
from sign_language_detection.exception.exception import CustomException

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights
from sklearn.metrics import f1_score

KINETICS_MEAN = np.array(
    [0.43216, 0.394666, 0.37645],
    dtype=np.float32
).reshape(3, 1, 1, 1)

KINETICS_STD = np.array(
    [0.22803, 0.22145, 0.21699],
    dtype=np.float32
).reshape(3, 1, 1, 1)
class AUTSLDataset(Dataset):

    def __init__(self, batch_files):

        self.samples = []

        for batch_path in batch_files:

            with np.load(batch_path) as data:

                labels = data["labels"]

                num_samples = len(labels)

                for index in range(num_samples):

                    self.samples.append(
                        (batch_path, index)
                    )

        if len(self.samples) == 0:
            raise ValueError(
                "Dataset contains no samples."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        batch_path, sample_index = self.samples[index]

        with np.load(batch_path) as data:

            features = data["features"][sample_index]
            label = data["labels"][sample_index]

        features = features.astype(
            np.float32
        )

        # (T, H, W, C) -> (C, T, H, W)
        features = np.transpose(
            features,
            (3, 0, 1, 2)
        )

        # Kinetics-400 normalization
        features = (
            features - KINETICS_MEAN
        ) / KINETICS_STD

        features = torch.from_numpy(
            features
        )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return features, label


class ModelTrainer:

    def __init__(self, config: ModelTrainerConfig):

        self.config = config

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        logger.info(
            f"Training device: {self.device}"
        )

        if self.device.type == "cuda":
            logger.info(
                f"GPU: {torch.cuda.get_device_name(0)}"
            )

    def _get_batch_files(self, split_name):

        try:

            transformed_data_dir = (
                self.config.transformed_data_dir
            )

            split_transformed_data_dir = (
                transformed_data_dir / split_name
            )

            if not split_transformed_data_dir.exists():

                raise Exception(
                    f"{split_transformed_data_dir} does not exist"
                )

            batch_files = sorted(
                split_transformed_data_dir.glob("*.npz")
            )

            if len(batch_files) == 0:

                raise Exception(
                    f"No npz file found in "
                    f"{split_transformed_data_dir}"
                )

            logger.info(
                f"Found {len(batch_files)} batch files "
                f"for {split_name}"
            )

            return batch_files

        except Exception as e:

            logger.error(
                f"Failed to get batch files for {split_name}"
            )

            raise CustomException(
                str(e),
                sys.exc_info()
            )

    def _create_dataloader(self, split_name):

        try:

            batch_files = self._get_batch_files(
                split_name
            )

            dataset = AUTSLDataset(
                batch_files
            )

            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=(split_name == "train"),
                num_workers=0,
                pin_memory=(self.device.type == "cuda")
            )

            logger.info(
                f"{split_name} dataset samples: "
                f"{len(dataset)}"
            )

            logger.info(
                f"{split_name} dataloader batches: "
                f"{len(dataloader)}"
            )

            return dataloader

        except Exception as e:

            logger.error(
                f"Failed to create dataloader "
                f"for {split_name}"
            )

            raise CustomException(
                str(e),
                sys.exc_info()
            )

    def _build_model(self):

        logger.info(
            "Loading R3D-18 Kinetics-400 pretrained model..."
        )

        weights = R3D_18_Weights.KINETICS400_V1

        model = r3d_18(
            weights=weights
        )

        in_features = model.fc.in_features

        model.fc = nn.Sequential(
            nn.Dropout(
                self.config.dropout_rate
            ),
            nn.Linear(
                in_features,
                self.config.num_classes
            )
        )

        # Freeze all parameters first

        for param in model.parameters():
            param.requires_grad = False

        # Fine-tune layer3

        for param in model.layer3.parameters():
            param.requires_grad = True

        # Fine-tune layer4

        for param in model.layer4.parameters():
            param.requires_grad = True

        # Fine-tune classifier

        for param in model.fc.parameters():
            param.requires_grad = True

        trainable_params = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        total_params = sum(
            p.numel()
            for p in model.parameters()
        )

        logger.info(
            f"Total parameters: {total_params:,}"
        )

        logger.info(
            f"Trainable parameters: "
            f"{trainable_params:,}"
        )

        model = model.to(self.device)

        return model

    def _compile_model(self, model):

        criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config.label_smoothing
        )

        optimizer = optim.AdamW(
            [
                {
                    "params": model.layer3.parameters(),
                    "lr": self.config.layer3_learning_rate
                },
                {
                    "params": model.layer4.parameters(),
                    "lr": self.config.layer4_learning_rate
                },
                {
                    "params": model.fc.parameters(),
                    "lr": self.config.fc_learning_rate
                },
            ],
            weight_decay=self.config.weight_decay
        )

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.config.scheduler_t0,
            T_mult=self.config.scheduler_t_mult,
            eta_min=self.config.scheduler_eta_min
        )

        return criterion, optimizer, scheduler

    def _train_one_epoch(
        self,
        model,
        dataloader,
        criterion,
        optimizer,
        scheduler,
        epoch
    ):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(
            dataloader
        ):

            inputs = inputs.to(
                self.device,
                non_blocking=True
            )

            labels = labels.to(
                self.device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(inputs)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                self.config.gradient_clip_value
            )

            optimizer.step()

            # Cosine warm restart is stepped
            # using fractional epoch progress.

            scheduler.step(
                epoch + (batch_idx + 1) / len(dataloader)
            )

            running_loss += (
                loss.item() * inputs.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                (predictions == labels)
                .sum()
                .item()
            )

            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_accuracy = correct / total

        return epoch_loss, epoch_accuracy

    @torch.no_grad()
    def _validate(
        self,
        model,
        dataloader,
        criterion
    ):

        model.eval()

        running_loss = 0.0

        all_labels = []
        all_predictions = []

        total = 0

        for inputs, labels in dataloader:

            inputs = inputs.to(
                self.device,
                non_blocking=True
            )

            labels = labels.to(
                self.device,
                non_blocking=True
            )

            outputs = model(inputs)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * inputs.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            total += labels.size(0)

        validation_loss = (
            running_loss / total
        )

        validation_accuracy = (
            np.mean(
                np.array(all_labels)
                == np.array(all_predictions)
            )
        )

        validation_macro_f1 = f1_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0
        )

        return (
            validation_loss,
            validation_accuracy,
            validation_macro_f1
        )

    def _save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        epoch,
        macro_f1
    ):

        checkpoint_dir = (
            self.config.checkpoint_dir
        )

        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "macro_f1": macro_f1,
            "num_classes": self.config.num_classes,
        }

        torch.save(
            checkpoint,
            self.config.checkpoint_file
        )

        logger.info(
            f"Best checkpoint saved: "
            f"{self.config.checkpoint_file}"
        )

    def _train_model(self):

        train_dataloader = (
            self._create_dataloader("train")
        )

        val_dataloader = (
            self._create_dataloader("val")
        )

        model = self._build_model()

        criterion, optimizer, scheduler = (
            self._compile_model(model)
        )

        best_macro_f1 = -1.0

        for epoch in range(
            self.config.epochs
        ):

            train_loss, train_accuracy = (
                self._train_one_epoch(
                    model,
                    train_dataloader,
                    criterion,
                    optimizer,
                    scheduler,
                    epoch
                )
            )

            (
                val_loss,
                val_accuracy,
                val_macro_f1
            ) = self._validate(
                model,
                val_dataloader,
                criterion
            )

            logger.info(
                f"Epoch [{epoch + 1}/"
                f"{self.config.epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.4f} | "
                f"Val Macro-F1: {val_macro_f1:.4f}"
            )

            if val_macro_f1 > best_macro_f1:

                best_macro_f1 = val_macro_f1

                self._save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    val_macro_f1
                )

        logger.info(
            f"Training completed. "
            f"Best Validation Macro-F1: "
            f"{best_macro_f1:.4f}"
        )

        # Load best checkpoint before returning model

        checkpoint = torch.load(
            self.config.checkpoint_file,
            map_location=self.device
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        return model

    def _save_model(self, model):

        model_dir = self.config.model_dir

        model_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        model_file = self.config.model_file

        torch.save(
            model.state_dict(),
            model_file
        )

        logger.info(
            f"Model saved successfully: "
            f"{model_file}"
        )

        return model_file

    def initiate_model_training(self):

        try:

            trained_model = self._train_model()

            model_file = self._save_model(
                trained_model
            )

            logger.info(
                "Model training completed successfully"
            )

            return model_file

        except Exception as e:

            logger.error(
                "Model training failed"
            )

            raise CustomException(
                str(e),
                sys.exc_info()
            )