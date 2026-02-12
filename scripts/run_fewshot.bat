@echo off
set PYTHONPATH=%cd%

echo Running Few-Shot Adaptation (EfficientNet-B5 + Pseudo 1x)...
python src/trainer/train.py ^
    --config configs/effnet_b5_fewshot.yaml ^
    --resume checkpoints/effnet_b5_kaggle_clean.pt

echo Done!
pause
