@echo off
set PYTHONPATH=%cd%

echo Running Inference with Kaggle EfficientNet-B5 (Clean Best)...
python src/trainer/inference.py ^
    --config configs/efficientnet_b5.yaml ^
    --checkpoint checkpoints/effnet_b5_kaggle_clean.pt ^
    --output data/submission/submission_effnet_b5_clean.csv ^
    --aggregation mean

echo Done! Output saved to data/submission/submission_effnet_b5_clean.csv
pause