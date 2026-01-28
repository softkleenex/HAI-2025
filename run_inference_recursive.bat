@echo off
set PYTHONPATH=%cd%

echo Running Inference with EfficientNet-B5 (Recursive Pseudo V10)...
python src/trainer/inference.py ^
    --config configs/effnet_b5_recursive.yaml ^
    --checkpoint checkpoints/effnet_b5_recursive_best.pt ^
    --output data/submission/submission_effnet_b5_recursive.csv ^
    --aggregation mean

echo Done! Output saved to data/submission/submission_effnet_b5_recursive.csv
pause
