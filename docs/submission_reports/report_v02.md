# Submission Report: Ensemble V2

## 1. Overview
*   **Submission ID:** 1394635
*   **Date:** 2026-01-19
*   **File Name:** `final_ensemble_v2.csv`
*   **Score (Public AUC):** **0.6781** (Best Baseline)

## 2. Strategy
*   **Objective:** Establish a strong baseline by combining diverse architectures trained on the provided external dataset (FaceForensics++).
*   **Method:** Weighted Ensemble of Top-3 Models.

## 3. Model Composition
| Model | Weight | Training Data | Description |
| :--- | :---: | :--- | :--- |
| **DINOv2 Large (Ep1)** | 0.45 | FF++ (140k) | Fine-tuned for 1 epoch. Excellent feature extractor. |
| **ViT Base** | 0.45 | FF++ (140k) | Standard Vision Transformer baseline. |
| **EfficientNet-B4** | 0.10 | FF++ (140k) | CNN based model for texture artifacts. |

## 4. Analysis
*   This was the first significant breakthrough, jumping from 0.56 (Single Model) to 0.67.
*   Proves that **diversity (ViT + CNN)** is key to generalization.
*   Used as the **Ground Truth (Teacher)** for subsequent Pseudo Labeling experiments.
