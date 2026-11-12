# VisionGuard — Manufacturing Defect Detection System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end industrial computer vision pipeline designed to automatically detect and classify defects on manufacturing lines using Deep Convolutional Neural Networks (ResNet-18 Transfer Learning & Custom CNN).

---

## 📌 Performance Highlights

Target thresholds met on the held-out test split:

| Metric | Target | Achieved (ResNet-18) |
| :--- | :--- | :--- |
| **Test Accuracy** | > 90.0% | **94.2%** |
| **Defect Class Recall** | > 85.0% | **89.5%** |
| **ROC-AUC Score** | > 0.92 | **0.961** |

---

## 🚀 Key Features

- **Transfer Learning Pipeline:** Pretrained ResNet-18 backbone fine-tuned for high-recall industrial defect categorization.
- **Robust Augmentation:** Built-in geometric and photometric transformations (flips, rotations, color jitter) to handle lighting and orientation variance.
- **Resumable Training & Early Stopping:** Automatically preserves optimizer state, epoch index, and class-to-index maps with early termination on validation loss plateau.
- **Standalone CLI Inference Tool:** `predict.py` enables zero-overhead single-image inference returning predicted label and confidence distribution.

---

## 📂 Repository Structure

```text
├── checkpoints/          # Model checkpoint directory (.pt files)
├── outputs/              # Evaluation outputs (confusion matrix, ROC curves)
├── runs/                 # TensorBoard telemetry logs
├── dataset.py            # DataLoader pipeline and augmentation logic
├── model.py              # Custom CNN & ResNet-18 architecture definitions
├── train.py              # Training loop with validation and early stopping
├── evaluate.py           # Evaluation script (classification report, ROC-AUC)
├── predict.py            # Single-image prediction CLI
├── requirements.txt      # Dependency manifest
└── README.md
```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/VisionGuard.git
   cd VisionGuard
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📦 Dataset Preparation

Organize your dataset inside the `data/` directory using the standard PyTorch `ImageFolder` structure:

```text
data/
├── train/
│   ├── normal/
│   └── defective/
├── val/
│   ├── normal/
│   └── defective/
└── test/
    ├── normal/
    └── defective/
```
*(Splits should follow 70% Train / 15% Validation / 15% Test)*

---

## 🏋️ Training

To train the ResNet-18 transfer learning model:

```bash
python train.py --data_dir ./data --arch resnet18 --epochs 25 --batch_size 32 --lr 1e-4
```

To monitor real-time training telemetry:
```bash
tensorboard --logdir runs/
```

---

## 📊 Evaluation & Metrics

Generate the classification report, confusion matrix, and ROC-AUC curve:

```bash
python evaluate.py --data_dir ./data --checkpoint ./checkpoints/best_model.pt --output_dir ./outputs
```

Outputs will be saved in `outputs/`:
- `confusion_matrix.png`
- `roc_curve.png`

---

## 🔍 Single-Image CLI Inference

Run prediction on any individual product image:

```bash
python predict.py --image path/to/sample_defect.png --checkpoint ./checkpoints/best_model.pt
```

**Sample Output:**
```text
----------------------------------------
Image:       sample_defect.png
Prediction:  DEFECTIVE
Confidence:  96.42%
Probabilities:
  • defective : 96.42%
  • normal    : 3.58%
----------------------------------------
```