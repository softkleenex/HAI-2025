@echo off
set PYTHONPATH=%cd%
python src/trainer/inference.py --config configs/dino_large.yaml --checkpoint checkpoints/dino_large_epoch1_snapshot.pt --output data/submission/submission_dino_large_ep1.csv > logs/inference_dino_ep1.log 2> logs/inference_dino_ep1_err.log
