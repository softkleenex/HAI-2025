# Submission Report: Ensemble V7

## 1. Overview
*   **Submission ID:** 1396350
*   **Date:** 2026-01-22
*   **File Name:** `final_ensemble_v7.csv`
*   **Score (Public AUC):** **0.7038** (The Breakthrough)

## 2. Strategy
*   **Objective:** Overcome the domain gap between FaceForensics++ and Dacon Test Set.
*   **Method:** **Pseudo Labeling (Test Data Injection)**.
*   **Technique:** "Full Injection" - Added 3,351 test images to the training set with labels predicted by V2.

## 3. Model Composition
| Model | Weight | Training Data | Description |
| :--- | :---: | :--- | :--- |
| **DINOv2 Large (Ep5)** | 0.50 | FF++ + **Test(Pseudo 1x)** | Trained for 5 epochs with test data mixed in (3% ratio). |
| **ViT Base** | 0.30 | FF++ | Stabilizer. |
| **EfficientNet-B4** | 0.20 | FF++ | Stabilizer. |

## 4. Analysis
*   **Key Finding:** Mixing a small amount (3%) of test data with Pseudo Labels significantly improved performance (+0.025).
*   This proved that the model can learn the "Test Distribution" even with imperfect labels.
