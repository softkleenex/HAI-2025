# 🕵️‍♂️ HAI 2025 Deepfake Detection Challenge Solution

> **Result:** Rank **121** / 1,356 (Top **9%**) | Private Score: **0.7040**  
> **Topic:** AI-generated Deepfake Video/Image Detection  
> **Host:** HAI (Hecto Financial) / Dacon

## 📌 Overview
This repository contains the solution for the [HAI 2025 Deepfake Detection Challenge](https://dacon.io/competitions/official/236628/overview/description).  
We employed a **Hybrid Ensemble Strategy** combining **Transformer (DINOv2)** and **CNN (EfficientNet, ConvNeXt)** architectures to tackle diverse deepfake generation techniques. Our key focus was on **Pseudo-Labeling (Semi-supervised Learning)** to adapt to the domain distribution of the test set.

## 🏆 Final Score
| Metric | Score | Rank |
|:---:|:---:|:---:|
| **Private AUC** | **0.70403** | **121st** (Top 9%) |
| Public AUC | 0.72151 | - |

---

## 🛠️ Solution Summary

### 1. Model Architecture (Ensemble Strategy)
We combined models with different inductive biases to maximize generalization performance.

| Model | Backbone | Role & Strategy | Weight |
|:---|:---|:---|:---:|
| **DINOv2 Large** | ViT (Self-supervised) | **[Main]** Captures global feature patterns. Uses ImageNet-21k pre-trained weights. | 0.5 |
| **EfficientNet-B5** | CNN | **[Adapter]** Detects local textures and artifacts. Fine-tuned with **Few-Shot Learning** on the test set. | 0.4 |
| **ConvNeXt Base** | CNN | **[Generalist]** Robust against blur and compression artifacts. Trained on clean FF++ data. | - |
| **ViT Base** | ViT | **[Support]** Ensures ensemble diversity. | 0.1 |

### 2. Key Techniques
*   **Pseudo-Labeling (Test Set Injection):**
    *   Inferred labels for the test set using an initial model and selected high-confidence samples (Confidence > 0.95 or < 0.05).
    *   Retrained the model with 10x oversampling of these pseudo-labeled samples.
    *   **Impact:** Public LB score improved from 0.67 to **0.72**.
*   **Test-Time Augmentation (TTA):**
    *   Applied `Horizontal Flip`, `Gaussian Blur`, and `Sharpen` during inference to stabilize predictions.
*   **Blur-Breaker Augmentation:**
    *   Used strong `ImageCompression` and `MotionBlur` during training to prevent misclassification of low-quality fake videos.

### 3. Data Strategy
*   **Training:** FaceForensics++ (140k images) + Pseudo-labeled Test Data (33k).
*   **Preprocessing:** Face cropping and alignment (224x224) using `facenet-pytorch` (MTCNN).
*   **Environment:**
    *   Local: NVIDIA RTX 4060 (8GB) - Inference & Lightweight Training.
    *   Cloud: Kaggle Notebooks (Dual T4 GPU) - Heavy Backbone Training.

---

## 📂 Project Structure
```text
├── checkpoints/       # Model weights (Not included in repo)
├── configs/           # YAML configs for experiments
├── data/              # Data directory (Raw data not included)
│   ├── submission/    # Inference Results
│   └── ...
├── scripts/
│   ├── run_kaggle.py  # Kaggle Automation Script
│   ├── ensemble.py    # Weighted Averaging Script
│   └── create_pseudo.py # Pseudo-label Generator
├── src/
│   ├── data/          # Dataset & Transforms
│   ├── models/        # DINO, EfficientNet, ConvNeXt Architectures
│   └── trainer/       # Training & Inference Loop
└── README.md
```

---

## 💻 Usage

### 1. Installation
```bash
git clone https://github.com/softkleenex/HAI-2025.git
cd HAI-2025
pip install -r requirements.txt
```

### 2. Inference (Local)
```bash
# Run Inference with TTA
python src/trainer/inference.py --config configs/dino_large.yaml --checkpoint checkpoints/dino_best.pt --output submission.csv --tta
```

### 3. Ensemble
```bash
# Combine multiple submissions
python scripts/ensemble.py --inputs sub1.csv sub2.csv --weights 0.5 0.5 --output final_ensemble.csv
```

---

## 📝 Retrospective & Experiments

| Experiment | Method | Result (Public) | Insight |
|:---:|:---|:---:|:---|
| **V2** | Baseline Ensemble (DINO+ViT+EffB4) | 0.6781 | Strong baseline performance with simple ensemble. |
| **V7** | **Pseudo Labeling (3% Injection)** | **0.7038** | Confirmed that domain adaptation is key. |
| **V10** | **Pseudo (10x) + FewShot** | **0.7215 (Best)** | Found optimal balance between overfitting and generalization. |
| **V15** | ConvNeXt (Blur Augmentation) | 0.7193 | Blur augmentation helped robustness but introduced label noise. |
| **V17** | Video Model (CNN+LSTM) | 0.7124 | Temporal features were less effective given the frame-based dataset. |

### 💡 What Worked
*   **Domain Adaptation:** Learning the distribution of the target data (Pseudo-Labeling) was far more effective than simply adding more external data (Celeb-DF).
*   **Strong Backbones:** Modern self-supervised models like DINOv2 acted as powerful feature extractors even with limited data.

### ⚠️ Challenges
*   **Video-Level Context:** Focusing on frame-level classification meant missing out on temporal inconsistencies.
*   **Resource Constraints:** Limited GPU memory prevented extensive experimentation with larger models (EfficientNet-B7, ViT-Large).

---

### 👨‍💻 Author
**Team softkleenex**
