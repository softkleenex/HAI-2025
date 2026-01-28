@echo off
set PYTHONPATH=%cd%

echo Running Weighted Ensemble V9 (Ultimate: DINO 10x + EffB5 Clean + ViT)...
python scripts/ensemble.py ^
    --inputs data/submission/submission_dino_10x_pseudo.csv data/submission/submission_effnet_b5_clean.csv data/submission/final_fixed_vit_submission.csv ^
    --weights 0.6 0.2 0.2 ^
    --output data/submission/final_ensemble_v9.csv

echo Done! Output saved to data/submission/final_ensemble_v9.csv
pause
