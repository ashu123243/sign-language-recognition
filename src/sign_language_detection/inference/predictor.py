from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18


# ============================================================
# KINETICS-400 NORMALIZATION
# Same normalization used during model training
# ============================================================

KINETICS_MEAN = np.array(
    [0.43216, 0.394666, 0.37645],
    dtype=np.float32
)

KINETICS_STD = np.array(
    [0.22803, 0.22145, 0.21699],
    dtype=np.float32
)


class SignLanguagePredictor:
    """
    Inference service for the trained AUTSL Sign Language
    Recognition model.
    """

    def __init__(
        self,
        model_path: str | Path,
        num_classes: int = 226,
        num_frames: int = 8,
        image_size: int = 160,
        dropout_rate: float = 0.5,
    ):
        self.model_path = Path(model_path)
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.image_size = image_size
        self.dropout_rate = dropout_rate

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"[INFO] Inference device: {self.device}")

        # ----------------------------------------------------
        # Build model
        # ----------------------------------------------------
        self.model = self._build_model()

        # ----------------------------------------------------
        # Load trained weights
        # ----------------------------------------------------
        self._load_model()

        # ----------------------------------------------------
        # Evaluation mode
        # ----------------------------------------------------
        self.model.eval()

        print("[INFO] Sign Language model loaded successfully.")

    # ========================================================
    # MODEL
    # ========================================================

    def _build_model(self):
        """
        Recreate the exact R3D-18 architecture used during
        training.
        """

        model = r3d_18(weights=None)

        in_features = model.fc.in_features

        model.fc = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(
                in_features,
                self.num_classes
            )
        )

        model = model.to(self.device)

        return model

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def _load_model(self):
        """
        Load the trained model state dictionary.
        """

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False
        )

        # ----------------------------------------------------
        # Final model contains only state_dict
        # ----------------------------------------------------

        if isinstance(checkpoint, dict):

            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]

                print(
                    "[INFO] Loading model_state_dict "
                    "from checkpoint."
                )

            else:
                state_dict = checkpoint

                print(
                    "[INFO] Loading model state dictionary."
                )

        else:
            raise ValueError(
                "Unsupported model file format."
            )

        self.model.load_state_dict(
            state_dict,
            strict=True
        )

    # ========================================================
    # FRAME SAMPLING
    # ========================================================

    def _sample_frame_indices(
        self,
        total_frames: int
    ):
        """
        Deterministic temporal sampling used for inference.

        Same basic strategy as validation/test:
        divide the video into segments and select the
        center frame from each segment.
        """

        if total_frames <= 0:
            return [0] * self.num_frames

        boundaries = np.linspace(
            0,
            total_frames,
            self.num_frames + 1
        ).astype(int)

        indices = []

        for i in range(self.num_frames):

            start = boundaries[i]
            end = boundaries[i + 1]

            if end > start:
                index = (start + end - 1) // 2
            else:
                index = min(
                    start,
                    total_frames - 1
                )

            index = max(
                0,
                min(index, total_frames - 1)
            )

            indices.append(index)

        return indices

    # ========================================================
    # BLACK CLIP
    # ========================================================

    def _make_black_clip(self):
        """
        Fallback clip if video decoding fails.
        """

        black = np.zeros(
            (
                self.num_frames,
                self.image_size,
                self.image_size,
                3
            ),
            dtype=np.float32
        )

        black = (
            black - KINETICS_MEAN
        ) / KINETICS_STD

        tensor = torch.from_numpy(black)

        tensor = tensor.permute(
            3, 0, 1, 2
        ).contiguous()

        return tensor.float()

    # ========================================================
    # VIDEO LOADING
    # ========================================================

    def _load_video_clip(
        self,
        video_path: str | Path
    ):
        """
        Load a video and convert it into:

        [3, T, H, W]

        where:
            T = 8 frames
            H = 160
            W = 160
        """

        video_path = str(video_path)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(
                f"[WARNING] Could not open video: "
                f"{video_path}"
            )

            return self._make_black_clip()

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if total_frames <= 0:
            print(
                f"[WARNING] Invalid frame count: "
                f"{video_path}"
            )

            cap.release()

            return self._make_black_clip()

        frame_indices = self._sample_frame_indices(
            total_frames
        )

        frames = []
        last_valid_frame = None

        try:

            for index in frame_indices:

                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    index
                )

                success, frame = cap.read()

                if not success:

                    if last_valid_frame is not None:
                        frame = last_valid_frame.copy()

                    else:
                        frame = np.zeros(
                            (
                                self.image_size,
                                self.image_size,
                                3
                            ),
                            dtype=np.uint8
                        )

                else:
                    last_valid_frame = frame.copy()

                # --------------------------------------------
                # BGR -> RGB
                # --------------------------------------------

                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                # --------------------------------------------
                # Resize
                # --------------------------------------------

                frame = cv2.resize(
                    frame,
                    (
                        self.image_size,
                        self.image_size
                    ),
                    interpolation=cv2.INTER_LINEAR
                )

                # --------------------------------------------
                # [0,255] -> [0,1]
                # --------------------------------------------

                frame = frame.astype(
                    np.float32
                ) / 255.0

                frames.append(frame)

        except Exception as error:

            print(
                f"[WARNING] Video processing failed: "
                f"{video_path}"
            )

            print(
                f"[WARNING] Error: {error}"
            )

            cap.release()

            return self._make_black_clip()

        finally:
            cap.release()

        # ----------------------------------------------------
        # Make sure we have exactly num_frames
        # ----------------------------------------------------

        if len(frames) != self.num_frames:

            while len(frames) < self.num_frames:

                if frames:
                    frames.append(
                        frames[-1].copy()
                    )
                else:
                    frames.append(
                        np.zeros(
                            (
                                self.image_size,
                                self.image_size,
                                3
                            ),
                            dtype=np.float32
                        )
                    )

            frames = frames[:self.num_frames]

        # ----------------------------------------------------
        # Stack
        # [T,H,W,C]
        # ----------------------------------------------------

        clip = np.stack(
            frames,
            axis=0
        )

        # ----------------------------------------------------
        # Kinetics normalization
        # ----------------------------------------------------

        clip = (
            clip - KINETICS_MEAN
        ) / KINETICS_STD

        # ----------------------------------------------------
        # NumPy -> Tensor
        # [T,H,W,C]
        # ->
        # [C,T,H,W]
        # ----------------------------------------------------

        clip = torch.from_numpy(
            clip
        ).float()

        clip = clip.permute(
            3, 0, 1, 2
        ).contiguous()

        return clip

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(
        self,
        video_path: str | Path,
        top_k: int = 5
    ):
        """
        Predict sign class from a video.

        Returns:
            {
                "predicted_class": int,
                "confidence": float,
                "top_predictions": [...]
            }
        """

        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video file not found: {video_path}"
            )

        # ----------------------------------------------------
        # Load video
        # ----------------------------------------------------

        clip = self._load_video_clip(
            video_path
        )

        # ----------------------------------------------------
        # Add batch dimension
        # [C,T,H,W]
        # ->
        # [1,C,T,H,W]
        # ----------------------------------------------------

        clip = clip.unsqueeze(0)

        clip = clip.to(self.device)

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        with torch.no_grad():

            logits = self.model(
                clip
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

        # ----------------------------------------------------
        # Top-K
        # ----------------------------------------------------

        k = min(
            top_k,
            self.num_classes
        )

        top_probs, top_indices = torch.topk(
            probabilities,
            k=k,
            dim=1
        )

        top_probs = top_probs[0].cpu().numpy()
        top_indices = top_indices[0].cpu().numpy()

        # ----------------------------------------------------
        # Main prediction
        # ----------------------------------------------------

        predicted_class = int(
            top_indices[0]
        )

        confidence = float(
            top_probs[0]
        )

        # ----------------------------------------------------
        # Top predictions
        # ----------------------------------------------------

        top_predictions = []

        for class_idx, probability in zip(
            top_indices,
            top_probs
        ):

            top_predictions.append(
                {
                    "class_id": int(class_idx),
                    "confidence": float(probability)
                }
            )

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top_predictions": top_predictions
        }


# ============================================================
# SIMPLE LOCAL TEST
# ============================================================

if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    MODEL_PATH = (
        PROJECT_ROOT
        / "artifacts"
        / "model"
        / "sign_language_model.pth"
    )

    predictor = SignLanguagePredictor(
        model_path=MODEL_PATH
    )

    print()
    print("=" * 60)
    print("MODEL LOADING TEST")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Device: {predictor.device}")
    print("Status: SUCCESS")
    print("=" * 60)