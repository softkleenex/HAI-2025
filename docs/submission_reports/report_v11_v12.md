# Submission Report: Ensemble V11 & V12

## 1. Overview
*   **V11:** `final_ensemble_v11.csv` (Score: 0.7027)
*   **V12:** `final_ensemble_v12.csv` (Score: Pending...)

## 2. Strategy (V11 - The Pure Pseudo)
*   **Hypothesis:** "If Pseudo Labeling is effective, we don't need external data (FF++) models anymore."
*   **Composition:** DINO 10x (50%) + EffB5 FewShot (50%).
*   **Result:** **Failed (0.7027).**
*   **Analysis:** Without a "Generalist" model (ViT trained on FF++), the ensemble suffered from **Confirmation Bias**. The models reinforced each other's errors on the Test Set.

## 3. Strategy (V12 - The Golden Ratio)
*   **Hypothesis:** "We need a balance between Test Set Specialists and Generalists."
*   **Composition:** 
    *   **DINO 10x (45%):** Main Specialist.
    *   **EffB5 FewShot (35%):** Secondary Specialist.
    *   **ViT Base (20%):** Generalist (increased from 10% in V10).
*   **Expectation:** Recovering diversity (ViT) should boost the score back above 0.72.
