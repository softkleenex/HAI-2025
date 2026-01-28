import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.trainer.inference import inference

if __name__ == "__main__":
    inference(
        config_path="configs/base_config.yaml",
        checkpoint_path="checkpoints/xception_strong_aug_best.pt",
        output_csv="data/submission/submission_xception.csv"
    )
