@echo off
set PYTHONPATH=%cd%

echo Running Inference with DINO Large (10x Pseudo Epoch 5) - MEAN Aggregation...
python src/trainer/inference.py ^
    --config configs/dino_large.yaml ^
    --checkpoint checkpoints/dino_large_finetune_best.pt ^
    --output data/submission/submission_dino_10x_pseudo.csv ^
    --aggregation mean

echo Done! Output saved to data/submission/submission_dino_10x_pseudo.csv
pause