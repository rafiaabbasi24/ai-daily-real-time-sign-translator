import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
from torch import nn

# ----------------------------------------------------------------------
# Configuration & Utilities
# ----------------------------------------------------------------------


def _setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )


def _load_model(model_path: Path, device: torch.device) -> nn.Module:
    """
    Load a TorchScript or a state‑dict model from ``model_path``.

    Parameters
    ----------
    model_path: Path
        Path to the serialized model file.
    device: torch.device
        Target device for inference.

    Returns
    -------
    nn.Module
        The model ready for evaluation.
    """
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        # Try loading as TorchScript (saved with torch.jit.save)
        model = torch.jit.load(str(model_path), map_location=device)
        logging.info("Loaded TorchScript model from %s", model_path)
    except RuntimeError:
        # Fallback to a state‑dict; the model class must be defined elsewhere.
        # For simplicity we assume a generic feed‑forward network architecture.
        # Users can replace ``SimpleMLP`` with their own architecture.
        class SimpleMLP(nn.Module):
            def __init__(self, input_dim: int = 63, hidden_dim: int = 128, num_classes: int = 26):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, num_classes),
                )

            def forward(self, x):
                return self.net(x)

        model = SimpleMLP()
        state = torch.load(str(model_path), map_location=device)
        model.load_state_dict(state)
        logging.info("Loaded state‑dict model from %s", model_path)

    model.eval()
    model.to(device)
    return model


def _preprocess_landmarks(landmarks: List[Tuple[float, float, float]]) -> torch.Tensor:
    """
    Convert normalized hand landmarks into a tensor suitable for the model.

    Parameters
    ----------
    landmarks: List[Tuple[float, float, float]]
        List of (x, y, z) normalized coordinates.

    Returns
    -------
    torch.Tensor
        Tensor of shape (1, 63) on CPU (moved to device later).
    """
    # Flatten the list and convert to float32 numpy array
    arr = np.array(landmarks, dtype=np.float32).reshape(1, -1)  # shape (1, 63)
    return torch.from_numpy(arr)


def _index_to_letter(idx: int) -> str:
    """Map a class index (0‑25) to the corresponding uppercase alphabet letter."""
    if 0 <= idx < 26:
        return chr(ord("A") + idx)
    return "?"


# ----------------------------------------------------------------------
# Core Translator
# ----------------------------------------------------------------------


class SignLanguageTranslator:
    """
    Real‑time sign‑language alphabet translator.

    The class encapsulates webcam handling, MediaPipe hand detection,
    preprocessing, model inference and result visualisation.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        confidence: float = 0.5,
    ) -> None:
        """
        Parameters
        ----------
        model: nn.Module
            Trained PyTorch model for alphabet classification.
        device: torch.device
            Device used for inference.
        confidence: float, optional
            Minimum detection confidence for MediaPipe Hands.
        """
        self.model = model
        self.device = device
        self.confidence = confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=self.confidence,
            min_tracking_confidence=self.confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.logger = logging.getLogger(self.__class__.__name__)

    def __del__(self):
        # Ensure MediaPipe resources are released
        self.hands.close()

    def _infer(self, landmarks: List[Tuple[float, float, float]]) -> str:
        """
        Run a forward pass on the pre‑processed landmarks.

        Returns
        -------
        str
            Predicted alphabet letter.
        """
        with torch.no_grad():
            tensor = _preprocess_landmarks(landmarks).to(self.device)
            logits = self.model(tensor)
            pred_idx = int(torch.argmax(logits, dim=1).cpu().item())
            return _index_to_letter(pred_idx)

    def run(self, camera_index: int = 0) -> None:
        """
        Start the webcam loop, perform detection and display results.

        Parameters
        ----------
        camera_index: int
            Index of the webcam (default 0).
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            self.logger.error("Cannot open camera with index %s", camera_index)
            return

        self.logger.info("Starting video capture (press 'q' to quit)")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Empty frame received; exiting loop")
                    break

                # Flip for a mirror view and convert to RGB for MediaPipe
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = self.hands.process(rgb_frame)

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]

                    # Draw landmarks on the frame
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2),
                    )

                    # Extract normalized (x, y, z) coordinates
                    normalized = [
                        (lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark
                    ]

                    # Inference
                    letter = self._infer(normalized)

                    # Overlay prediction
                    cv2.putText(
                        frame,
                        f"Predicted: {letter}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.putText(
                        frame,
                        "No hand detected",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow("Sign Language Alphabet Translator", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.logger.info("Quit signal received")
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.logger.info("Resources released; exiting.")


# ----------------------------------------------------------------------
# CLI Entrypoint
# ----------------------------------------------------------------------


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(
        description="Real‑time Sign Language Alphabet Translator"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the TorchScript (.pt) or state‑dict model file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device (e.g., 'cpu' or 'cuda').",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Index of the webcam to use.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence for MediaPipe Hands.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Application entry point."""
    args = parse_args()
    _setup_logging(level=getattr(logging, args.log_level.upper()))

    device = torch.device(args.device)
    model = _load_model(args.model_path, device)

    translator = SignLanguageTranslator(
        model=model,
        device=device,
        confidence=args.confidence,
    )
    translator.run(camera_index=args.camera_index)


if __name__ == "__main__":
    main()
