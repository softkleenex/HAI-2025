import pandas as pd
import os

def fix_and_ensemble():
    # 1. Load Data
    sample_df = pd.read_csv('data/submission/sample_submission.csv') # Target (500)
    
    dino_df = pd.read_csv('data/submission/submission_dino_10x_pseudo.csv')
    eff_df = pd.read_csv('data/submission/submission_effnet_b5_fewshot.csv')
    conv_df = pd.read_csv('data/submission/submission_convnext_tta.csv') # Source (505 - dirty)
    
    # 2. Map ConvNeXt probs to Sample Filenames
    # Need to handle extension mismatch: 'TEST_001.mp4' -> 'TEST_001.jpg'
    # Strategy: Strip extension and map based on ID (TEST_XXX)
    
    conv_df['id'] = conv_df['filename'].apply(lambda x: os.path.splitext(x)[0])
    sample_df['id'] = sample_df['filename'].apply(lambda x: os.path.splitext(x)[0])
    
    # Merge (Left Join on ID)
    merged = sample_df.merge(conv_df[['id', 'prob']], on='id', how='left', suffixes=('', '_conv'))
    
    # Check for NaNs
    if merged['prob_conv'].isnull().sum() > 0:
        print("⚠️ Warning: Some files missing in ConvNeXt results!")
        # Fill with mean or something? No, let's check first.
    
    # 3. Align other models (Assumed sorted, but safer to merge)
    dino_df['id'] = dino_df['filename'].apply(lambda x: os.path.splitext(x)[0])
    eff_df['id'] = eff_df['filename'].apply(lambda x: os.path.splitext(x)[0])
    
    merged = merged.merge(dino_df[['id', 'prob']], on='id', how='left', suffixes=('', '_dino'))
    merged = merged.merge(eff_df[['id', 'prob']], on='id', how='left', suffixes=('', '_eff'))
    
    # 4. Calculate Ensemble
    # Weights: DINO(0.5) + ConvNeXt(0.3) + EffB5(0.2)
    # Column names: prob_conv, prob_dino, prob_eff
    
    merged['final_prob'] = (merged['prob_dino'] * 0.5) + \
                           (merged['prob_conv'] * 0.3) + \
                           (merged['prob_eff'] * 0.2)
                           
    # 5. Create Final Submission
    submission = merged[['filename', 'final_prob']].rename(columns={'final_prob': 'prob'})
    
    output_path = 'data/submission/final_ensemble_v15_fixed.csv'
    submission.to_csv(output_path, index=False)
    print(f"✅ Ensemble V15 (Fixed & Aligned) Saved to: {output_path}")
    print(f"Shape: {submission.shape}")
    print(submission.head())

if __name__ == '__main__':
    fix_and_ensemble()