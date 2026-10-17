# Real-Time Sign Language Alphabet Translator

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Demo](#demo)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview
The **Real-Time Sign Language Alphabet Translator** is a deep learning application that captures hand gestures from a webcam, interprets them using a hybrid vision‑transformer model, and outputs the corresponding English alphabet letters in real time. The system combines MediaPipe for robust hand landmark detection, OpenCV for video handling, and a custom transformer‑based classifier built with PyTorch.

## Features
- **Live video capture** with low latency (< 30 ms per frame on a modern GPU).
- **Hand landmark extraction** using MediaPipe Hands (21 3‑D keypoints per hand).
- **Hybrid CNN‑Transformer architecture** for high‑accuracy alphabet classification.
- **Pre‑trained model** (97.8 % accuracy on the ASL Alphabet dataset) ready for inference.
- **Easy-to‑extend** codebase for adding new gestures or languages.
- **Cross‑platform** (Linux, macOS, Windows) with only Python dependencies.

## Demo
```bash
python src/main.py --source 0 --model checkpoints/asl_transformer.pt
```
A webcam window will open. Show an ASL alphabet sign to the camera and the predicted letter will appear in the top‑left corner.

## Architecture
```
Video Frame (640×480) ──► OpenCV Capture
        │
        ▼
MediaPipe Hands (21 landmarks) ──► Normalization
        │
        ▼
CNN Backbone (ResNet‑18) ──► Feature Map (512‑dim)
        │
        ▼
Transformer Encoder (4 layers, 8 heads) ──► Sequence Embedding
        │
        ▼
Linear Classifier (26 classes) ──► Softmax → Predicted Letter
```

- **CNN Backbone** extracts spatial features from the hand region.
- **Transformer Encoder** captures temporal dependencies when processing a short sliding window of frames (default: 5 frames).
- **Linear Classifier** maps the final embedding to one of the 26 alphabet classes.

## Installation

### Prerequisites
- Python **3.9** or newer
- **CUDA** 11.7+ (optional, for GPU acceleration)
- `git` (to clone the repository)

### Step‑by‑step

```bash
# Clone the repository
git clone https://github.com/yourusername/real-time-sign-language-translator.git
cd real-time-sign-language-translator

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Upgrade pip and install core dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

#### `requirements.txt` (for reference)
```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
mediapipe>=0.10.0
transformers>=4.35.0
numpy>=1.24.0
tqdm>=4.66.0
pillow>=10.0.0
```

If you plan to train on a GPU, ensure the correct PyTorch build:

```bash
# Example for CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Configuration
All configurable parameters are stored in `config.yaml`. Common options:

```yaml
video:
  source: 0                # Webcam index or video file path
  width: 640
  height: 480
  fps: 30

model:
  architecture: transformer
  backbone: resnet18
  num_layers: 4
  num_heads: 8
  pretrained_path: checkpoints/asl_transformer.pt

inference:
  sliding_window: 5        # Number of frames for temporal context
  confidence_threshold: 0.85
```

You can override any option via command‑line arguments (see Usage).

## Usage

### Real‑time Inference
```bash
python src/main.py \
    --config config.yaml \
    --source 0 \
    --model checkpoints/asl_transformer.pt \
    --window 5 \
    --threshold 0.85
```

#### Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to a YAML configuration file | `config.yaml` |
| `--source` | Video source (`0` for default webcam, or path to video file) | `0` |
| `--model` | Path to the trained `.pt` checkpoint | `checkpoints/asl_transformer.pt` |
| `--window` | Number of frames used for temporal smoothing | `5` |
| `--threshold` | Minimum confidence to display a prediction | `0.85` |
| `--no-display` | Run inference without opening a GUI window (useful for headless servers) | — |

### Batch Evaluation
```bash
python src/evaluate.py \
    --data data/test/ \
    --model checkpoints/asl_transformer.pt \
    --batch-size 64
```
The script prints overall accuracy, per‑class recall, and a confusion matrix plot saved to `outputs/`.

## Training the Model

### Dataset
The project uses the **ASL Alphabet** dataset (26 classes, 3000+ images per class). Download it and place it under `data/`.

```bash
# Example download script
bash scripts/download_dataset.sh
```

### Training Script
```bash
python src/train.py \
    --config config.yaml \
    --data-dir data/train/ \
    --epochs 30 \
    --batch-size 128 \
    --lr 3e-4 \
    --output checkpoints/
```

Key options:
- `--epochs`: Number of training epochs.
- `--lr`: Learning rate for AdamW optimizer.
- `--output`: Directory where checkpoints and logs will be saved.

Training logs are written to TensorBoard. Launch with:

```bash
tensorboard --logdir runs/
```

## Evaluation
The `src/evaluate.py` script computes standard metrics:

```bash
python src/evaluate.py \
    --data data/val/ \
    --model checkpoints/asl_transformer.pt \
    --batch-size 256 \
    --output outputs/
```

Outputs:
- `metrics.json` – accuracy, precision, recall, F1.
- `confusion_matrix.png` – visual confusion matrix.
- `predictions.csv` – per‑sample predictions.

## Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Ensure code style compliance (`flake8 .` and `black .`).
4. Add or update tests under `tests/`.
5. Run the full test suite: `pytest -q`.
6. Submit a Pull Request with a clear description of changes.

### Code Style
- Use **PEP 8** conventions.
- Type hints are required for all public functions.
- Docstrings should follow the **Google** style.

## License
This project is licensed under the **MIT License**. See the `LICENSE` file for details.

## Acknowledgements
- **MediaPipe** for reliable hand landmark detection.
- **PyTorch** for the deep learning framework.
- The **ASL Alphabet** dataset creators.
- Open‑source contributors of the libraries used in this project.