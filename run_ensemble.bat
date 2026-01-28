@echo off
set PYTHONPATH=%cd%

echo Running Weighted Ensemble V14 (DINO 10x + EffB5 Recursive V2 + ViT)...
python scripts/ensemble.py ^
    --inputs data/submission/submission_dino_10x_pseudo.csv data/submission/submission_effnet_b5_recursive.csv data/submission/final_fixed_vit_submission.csv ^
    --weights 0.5 0.4 0.1 ^
    --output data/submission/final_ensemble_v14.csv

echo Done! Output saved to data/submission/final_ensemble_v14.csv
pause
