# HAI 2025 Deepfake Detection Challenge

## 🏆 Current Status
*   **Best Score (Public):** **0.7215** (Ensemble V10)
*   **Rank:** Mid-Tier (Target: 0.9+)
*   **Best Strategy:** Pseudo Labeling (Test Set Injection) + Ensemble

## 🛠️ Model Architecture
1.  **DINOv2 Large (ViT):**
    *   Pre-trained on ImageNet-21k (Self-supervised)
    *   Fine-tuned on FF++ (100k) + Test Set (33k Pseudo-labeled)
    *   Key Strength: Global feature extraction & High confidence
2.  **EfficientNet-B5 (CNN):**
    *   Trained on Kaggle T4 x2 (Clean FF++ 140k)
    *   Fine-tuned on Test Set Only (Few-Shot Adaptation)
    *   Key Strength: Texture analysis & Regularization
3.  **ViT Base:**
    *   Baseline Model
    *   Key Strength: Ensemble Diversity

## 📂 Project Structure
```text
/
├── checkpoints/       # Trained Models (.pt)
├── configs/           # Experiment Configs (.yaml)
├── data/
│   ├── raw/           # FF++ Dataset (140k)
│   ├── processed/     # Processed Test Images
│   └── submission/    # Submission CSVs
├── docs/              # Documentation & History
│   ├── SUBMISSION_HISTORY.csv  # ★ All Records
│   └── submission_reports/     # Detailed Reports
├── logs/              # Training Logs
├── scripts/           # Utility Scripts (Ensemble, Pseudo-labeling)
└── src/               # Source Code (Model, Trainer, Dataset)
```

## 🚀 Key Experiments
| Exp | Method | Result | Note |
|:---:|:---|:---:|:---|
| **V2** | Ensemble (DINO Ep1 + ViT + EffB4) | 0.6781 | Strong Baseline |
| **V7** | **Pseudo Labeling (3% Injection)** | **0.7038** | Proof of Concept |
| **V10** | **Pseudo (10x) + FewShot Adaptation** | **0.7215** | Current Best |
| **V11** | Pure Pseudo (No External Data) | 0.7027 | Failed (Need Diversity) |

## 📅 Next Plan (To 0.9)
*   **Strict Pseudo Labeling:** Filter only high-confidence samples (0.05 / 0.95)
*   **Clean Fine-tuning:** Retrain DINO from Epoch 1 with strict labels.
*   **Advanced Ensemble:** Combine strict models with generalists.