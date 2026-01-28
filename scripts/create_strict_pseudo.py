import pandas as pd
import os
import glob

def create_strict_pseudo_labels(input_csv, threshold_real=0.05, threshold_fake=0.95):
    print(f"🔍 Reading predictions from {input_csv}...")
    if not os.path.exists(input_csv):
        print("❌ Error: Input CSV not found!")
        return

    df = pd.read_csv(input_csv)
    
    # Strict Filtering
    real_samples = df[df['prob'] < threshold_real].copy()
    fake_samples = df[df['prob'] > threshold_fake].copy()
    
    real_samples['label'] = 0
    fake_samples['label'] = 1
    
    pseudo_df = pd.concat([real_samples, fake_samples])
    
    print(f"Total Samples: {len(df)}")
    print(f"✅ Strict Real (<{threshold_real}): {len(real_samples)}")
    print(f"✅ Strict Fake (>{threshold_fake}): {len(fake_samples)}")
    print(f"Selected Total: {len(pseudo_df)} ({len(pseudo_df)/len(df)*100:.1f}%)")
    
    # Image Mapping
    processed_dir = os.path.abspath('data/processed/test')
    valid_rows = []
    
    print(f"📂 Scanning images in {processed_dir}...")
    for idx, row in pseudo_df.iterrows():
        filename = row['filename']
        file_base = os.path.splitext(filename)[0]
        video_folder = os.path.join(processed_dir, file_base)
        
        if os.path.isdir(video_folder):
            frames = glob.glob(os.path.join(video_folder, '*.jpg'))
            for frame_path in frames:
                valid_rows.append({'path': frame_path, 'label': row['label']})
    
    # 10x Oversampling (Confidence is high, so boost it!)
    if len(valid_rows) > 0:
        final_df = pd.DataFrame(valid_rows)
        # Apply 10x Oversampling directly
        final_df = pd.concat([final_df]*10, ignore_index=True)
        
        output_path = 'data/pseudo_train.csv'
        final_df.to_csv(output_path, index=False)
        print(f"💾 Saved {len(final_df)} samples (10x Oversampled) to {output_path}")
    else:
        print("⚠️ No samples selected! Threshold might be too strict.")

if __name__ == "__main__":
    create_strict_pseudo_labels('data/submission/submission_dino_10x_pseudo.csv')
