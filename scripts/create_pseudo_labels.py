import pandas as pd
import os
import shutil

import glob
import pandas as pd
import os

def create_pseudo_labels(input_csv='data/submission/final_ensemble_v10.csv', threshold_real=0.2, threshold_fake=0.8):
    df = pd.read_csv(input_csv)
    
    # --- Recursive Pseudo Labeling (Aggressive) ---
    real_samples = df[df['prob'] < threshold_real].copy()
    fake_samples = df[df['prob'] > threshold_fake].copy()
    
    real_samples['label'] = 0
    fake_samples['label'] = 1
    
    pseudo_df = pd.concat([real_samples, fake_samples])
    
    print(f"Total Test Samples: {len(df)}")
    print(f"Selected Real (Label 0): {len(real_samples)}")
    print(f"Selected Fake (Label 1): {len(fake_samples)}")
    print(f"Total Pseudo Videos: {len(pseudo_df)} ({len(pseudo_df)/len(df)*100:.1f}%)")
    
    valid_rows = []
    processed_dir = os.path.abspath('data/processed/test') 
    
    print(f"Scanning images in: {processed_dir}")

    for idx, row in pseudo_df.iterrows():
        filename = row['filename']
        file_base = os.path.splitext(filename)[0]
        
        video_folder = os.path.join(processed_dir, file_base)
        
        if os.path.isdir(video_folder):
            frames = glob.glob(os.path.join(video_folder, '*.jpg'))
            for frame_path in frames:
                valid_rows.append({'path': frame_path, 'label': row['label']})
        
    final_pseudo_df = pd.DataFrame(valid_rows)
    output_path = 'data/pseudo_train.csv'
    final_pseudo_df.to_csv(output_path, index=False)
    print(f"Saved {len(final_pseudo_df)} valid pseudo-labeled frames (Recursive V10) to {output_path}")

if __name__ == "__main__":
    create_pseudo_labels()

if __name__ == "__main__":
    create_pseudo_labels('data/submission/final_ensemble_v2.csv')