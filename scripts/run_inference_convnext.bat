@echo off
set PYTHONPATH=%cd%

echo Running Inference with ConvNeXt (Kaggle Trained) + TTA...
python src/trainer/inference.py ^
    --config configs/convnext_inference.yaml ^
    --checkpoint checkpoints/convnext_kaggle_best.pt ^
    --output data/submission/submission_convnext_tta.csv ^
    --tta

echo Done! Output saved to data/submission/submission_convnext_tta.csv
pause
