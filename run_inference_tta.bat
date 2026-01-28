@echo off
set PYTHONPATH=%cd%

echo Running Inference with TTA (Test Time Augmentation)...
python src/trainer/inference.py ^
    --config configs/dino_large.yaml ^
    --checkpoint checkpoints/dino_large_finetune_best.pt ^
    --output data/submission/submission_dino_tta.csv ^
    --aggregation mean ^
    --tta

echo Done! Output saved to data/submission/submission_dino_tta.csv
pause
