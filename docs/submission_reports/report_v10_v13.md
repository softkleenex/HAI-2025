# Submission Report: Ensemble V10 - V13 (Pseudo Labeling Experiments)

## 1. Overview
*   **V10 (Best):** 0.7215 (DINO 0.5 / EffB5 0.4 / ViT 0.1)
*   **V11:** 0.7027 (Pure Pseudo - No ViT)
*   **V13:** 0.7101 (Balanced Pseudo - DINO 0.45 / EffB5 0.45)

## 2. Key Insights
1.  **DINO Dominance:** DINO Large (Pseudo 10x) is the strongest model. Increasing its weight improves the score.
2.  **Diversity Matters:** Removing ViT (V11) caused a performance drop (-0.02). Even a weak model (ViT) helps generalization.
3.  **Efficiency Limit:** Adjusting weights only yields marginal gains (0.01 range). To reach 0.9, we need **better features**, not just better blending.

## 3. Conclusion
*   **Optimal Mix:** DINO > EffB5 > ViT.
*   **Current Ceiling:** ~0.72.
*   **Requirement for 0.9:** Need to fundamentally change the **Test Set Adaptation** strategy.
