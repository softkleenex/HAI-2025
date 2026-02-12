@echo off
set PYTHONPATH=%cd%
python src/trainer/train.py --config configs/dino_large.yaml --resume checkpoints/dino_large_epoch1_snapshot.pt > logs/train_dino_resume.log 2> logs/train_dino_resume_err.log
