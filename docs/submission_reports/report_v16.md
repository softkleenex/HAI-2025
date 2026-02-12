# Submission Report: Ensemble V16 (Celeb-DF Injection)

## 1. Overview
*   **Submission ID:** (Pending)
*   **Date:** 2026-01-31
*   **File Name:** `final_ensemble_v16.csv`
*   **Score:** (Pending)

## 2. Strategy
*   **Objective:** Enhance generalization by introducing a high-quality external dataset (Celeb-DF).
*   **Method:** Trained `ConvNeXt-Base` on combined FF++ and Celeb-DF datasets with strong quality augmentations (Blur-Breaker).

## 3. Model Composition
| Model | Weight | Training Data | Description |
| :--- | :---: | :--- | :--- |
| **DINO Large (10x)** | 0.50 | FF++ + Test(Pseudo 10x) | Test Set Specialist. |
| **ConvNeXt Celeb** | 0.30 | FF++ + Celeb-DF | Generalist with broad knowledge. |
| **EffB5 Recursive** | 0.20 | Test(Pseudo 1x) Only | Fine-tuned adapter. |

## 4. Hypothesis
*   Celeb-DF features will help distinguish "real high-quality" from "high-quality fake", complementing DINO's pattern matching.
