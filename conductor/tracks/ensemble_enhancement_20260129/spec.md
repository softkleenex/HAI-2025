# Track Specification: Ensemble Enhancement

## 1. Goal
Improve the Leaderboard score (target 0.75+) by integrating a high-performance `ConvNeXt-Base` model trained on Kaggle into the existing ensemble and implementing Test-Time Augmentation (TTA) to handle image quality variations.

## 2. Core Features
- **ConvNeXt Training on Kaggle:** Train `convnext_base` on the `140k-real-and-fake-faces` dataset using Kaggle's dual T4 GPUs.
- **Robustness Augmentation:** Apply strong blur and compression augmentations during training ("Blur-Breaker") to improve generalization on the test set.
- **TTA Inference:** Implement TTA (Flip, Sharpen, Blur, Compression) in the inference pipeline to stabilize predictions.
- **Ensemble Integration:** Update the ensemble logic to include the new ConvNeXt model with optimized weights.

## 3. Tech Stack
- **Framework:** PyTorch, Timm
- **Augmentation:** Albumentations
- **Environment:** Kaggle Notebooks (Training), Local PC (Inference & Ensemble)

## 4. Success Criteria
- **Model Accuracy:** ConvNeXt model achieves >99% validation accuracy on the 140k dataset.
- **TTA Functionality:** Inference script supports `--tta` flag and correctly averages predictions from augmented inputs.
- **Leaderboard Score:** The new ensemble (V15) achieves a higher public AUC score than the previous best (V10: 0.7215).