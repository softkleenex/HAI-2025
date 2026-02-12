@echo off
set PYTHONPATH=%cd%

echo Running Weighted Ensemble V17 (DINO 10x + Video Model Fixed + EffB5 FewShot)...
python scripts/ensemble.py ^
    --inputs data/submission/submission_dino_10x_pseudo.csv data/submission/submission_video_fixed.csv data/submission/submission_effnet_b5_fewshot.csv ^
    --weights 0.5 0.3 0.2 ^
    --output data/submission/final_ensemble_v17.csv

echo Done! Output saved to data/submission/final_ensemble_v17.csv
pause
