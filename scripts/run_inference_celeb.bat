@echo off
set PYTHONPATH=%cd%

echo Running Inference with ConvNeXt (Celeb-DF Generalist) + TTA...
python src/trainer/inference.py ^
    --config configs/convnext_celeb.yaml ^
    --checkpoint checkpoints/convnext_generalist_best.pt ^
    --output data/submission/submission_convnext_celeb.csv ^
    --tta

echo Done! Output saved to data/submission/submission_convnext_celeb.csv
pause
