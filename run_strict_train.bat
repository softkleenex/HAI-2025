@echo off
set PYTHONPATH=%cd%

echo Running DINO Large Strict Training (Epoch 1 -> 3)...
python src/trainer/train.py ^
    --config configs/dino_large.yaml ^
    --resume checkpoints/dino_large_epoch1_snapshot.pt

echo Done!
pause
