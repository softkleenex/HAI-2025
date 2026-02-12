import pandas as pd
import os
import glob

def fix_all_submissions():
    # 1. Load Sample (Correct Template)
    sample_path = 'data/submission/sample_submission.csv'
    if not os.path.exists(sample_path):
        print("❌ Sample submission not found!")
        return
        
    sample_df = pd.read_csv(sample_path)
    sample_df['id'] = sample_df['filename'].apply(lambda x: os.path.splitext(x)[0])
    
    print(f"✅ Loaded Sample: {len(sample_df)} rows")

    # 2. Find all submission files
    target_files = [
        'data/submission/submission_dino_10x_pseudo.csv',
        'data/submission/submission_effnet_b5_recursive.csv',
        'data/submission/submission_convnext_celeb.csv',
        'data/submission/submission_dino_large_best_mean.csv' # DINO Ep2
    ]
    
    for f in target_files:
        if not os.path.exists(f):
            print(f"⚠️ File not found: {f}")
            continue
            
        print(f"\n🔧 Fixing {f}...")
        df = pd.read_csv(f)
        
        # Check if already fixed (filename matches sample)
        if len(df) == len(sample_df) and df['filename'].equals(sample_df['filename']):
            print("   -> Already correct format. Skipping.")
            continue
            
        # Fix Logic: Map by ID
        df['id'] = df['filename'].apply(lambda x: os.path.splitext(x)[0])
        
        # Merge
        merged = sample_df[['filename', 'id']].merge(df[['id', 'prob']], on='id', how='left')
        
        # Fill NaNs
        nans = merged['prob'].isnull().sum()
        if nans > 0:
            print(f"   ⚠️ Warning: {nans} missing predictions. Filling with 0.5")
            merged['prob'] = merged['prob'].fillna(0.5)
            
        # Save as _fixed.csv
        new_path = f.replace('.csv', '_fixed.csv')
        merged[['filename', 'prob']].to_csv(new_path, index=False)
        print(f"   ✅ Saved fixed file to: {new_path}")

if __name__ == '__main__':
    fix_all_submissions()
