@echo off
set PYTHONPATH=%cd%

echo Running Inference with EfficientNet-B5 (Few-Shot Pseudo)...
python src/trainer/inference.py ^
    --config configs/effnet_b5_fewshot.yaml ^
    --checkpoint checkpoints/effnet_b5_fewshot_best.pt ^
    --output data/submission/submission_effnet_b5_fewshot.csv ^
    --aggregation mean

echo Done! Output saved to data/submission/submission_effnet_b5_fewshot.csv
pause
