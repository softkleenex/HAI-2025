# Submission Report: Ensemble V10

## 1. Overview
*   **Submission ID:** 1396850
*   **Date:** 2026-01-23
*   **File Name:** `final_ensemble_v10.csv`
*   **Score (Public AUC):** **0.7215** (Current Best)

## 2. Strategy
*   **Objective:** Maximize Test Set adaptation using multiple Pseudo-Labeling techniques.
*   **Method:** **Iterative Pseudo Labeling** + **Test-Time Fine-tuning**.

## 3. Model Composition
| Model | Weight | Training Data | Description |
| :--- | :---: | :--- | :--- |
| **DINO Large (10x)** | 0.50 | FF++ + **Test(Pseudo 10x)** | Aggressive oversampling of test data (25% ratio). |
| **EffB5 FewShot** | 0.40 | **Test(Pseudo 1x) ONLY** | Fine-tuned *only* on test data for 2 epochs. (Domain Adaptation). |
| **ViT Base** | 0.10 | FF++ | Minimal regularization to prevent mode collapse. |

## 4. Analysis
*   **Success:** Combining "Oversampling" (DINO) and "Few-Shot Adaptation" (EffB5) yielded the best result.
*   The "EffB5 FewShot" model likely acted as a **Test Set Specialist**.
*   Proved that aggressive adaptation works better than conservative mixing.
