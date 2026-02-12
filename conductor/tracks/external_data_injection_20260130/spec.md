# Track Specification: External Data Injection

## 1. Goal
Break the score plateau (0.72) by introducing a new, high-quality dataset (**Celeb-DF**) into the training pipeline. This aims to prevent overfitting to FaceForensics++ (FF++) artifacts and improve the model's ability to detect more realistic deepfakes.

## 2. Core Features
- **Dataset Acquisition:** Locate and use the `Celeb-DF` (v2) dataset on Kaggle.
- **Data Integration:** Update the `DeepFakeDataset` class to handle Celeb-DF's folder structure and combine it with FF++.
- **Model Training:** Retrain `ConvNeXt` (or `EfficientNet-V2`) on the combined dataset (FF++ + Celeb-DF).
- **Ensemble V16:** Create a new ensemble incorporating this "Generalist" model.

## 3. Tech Stack
- **Kaggle Notebook:** For downloading and training on large external datasets.
- **PyTorch/Timm:** For model training.

## 4. Success Criteria
- **Validation Accuracy:** Maintain >98% accuracy on the combined validation set.
- **Leaderboard Score:** Surpass 0.7215 with Ensemble V16.
