# Product Definition

## Vision
To achieve a top-tier ranking (0.9+ AUC) in the HAI 2025 Deepfake Detection Challenge by developing a highly accurate and robust deepfake detection system. The project focuses purely on maximizing the competition metric through advanced model architectures, ensemble strategies, and data-centric approaches.

## Target Audience
- **Competition Organizers (Hecto Financial):** The primary stakeholder evaluating the model's performance.
- **Data Scientists/Engineers:** The team developing and refining the models.

## Key Features
- **Competition-Optimized Pipeline:** A streamlined workflow designed specifically to maximize the Leaderboard score.
- **Advanced Ensemble Framework:** Utilization of diverse models (DINOv2, EfficientNet, ViT, ConvNeXt) and sophisticated combination techniques (Weighted Average, Stacking).
- **Pseudo-Labeling Strategy:** Implementation of iterative self-training and semi-supervised learning using test data to adapt to the target domain.
- **Test-Time Augmentation (TTA):** Robust inference using various augmentations (flip, blur, compression) to handle image quality variations.
- **Hybrid Training Environment:** Leveraging both local resources (RTX 4060) and cloud resources (Kaggle T4 x2) for efficient model training.

## Success Metrics
- **Primary Metric:** AUC (Area Under the Receiver Operating Characteristic Curve) on the Private Leaderboard.
- **Target Score:** 0.9+ AUC.
