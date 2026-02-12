@echo off
set PYTHONPATH=%cd%

echo Running Recursive Few-Shot Adaptation (V10 Base)...
python src/trainer/train.py ^
    --config configs/effnet_b5_recursive.yaml ^
    --resume checkpoints/effnet_b5_kaggle_clean.pt

echo Done!
pause
