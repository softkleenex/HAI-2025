import pandas as pd
import sys

def fix_submission_strict(sample_path, target_path, output_path):
    # Load
    sample = pd.read_csv(sample_path)
    target = pd.read_csv(target_path)
    
    # Check shape
    print(f"Sample: {sample.shape}, Target: {target.shape}")
    
    # Map predictions to sample order
    # Create dictionary for fast lookup
    pred_dict = dict(zip(target['filename'], target['prob']))
    
    # Fill probabilities ensuring order matches sample
    # If a filename is missing in target, use 0.5
    sample['prob'] = sample['filename'].map(pred_dict).fillna(0.5)
    
    # Check for NaNs again
    if sample['prob'].isna().sum() > 0:
        print("Warning: Still NaNs found!")
    
    # Save
    sample.to_csv(output_path, index=False)
    print(f"Strictly fixed submission saved to {output_path}")
    print(sample.head())

if __name__ == "__main__":
    fix_submission_strict(
        'data/submission/sample_submission.csv',
        'data/submission/final_dino_large_epoch1.csv',
        'data/submission/final_dino_large_epoch1_fixed.csv'
    )
