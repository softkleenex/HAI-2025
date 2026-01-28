import pandas as pd
import os
import argparse

def fix_submission(sample_path, my_path, output_path):
    if not os.path.exists(sample_path):
        print(f"Error: {sample_path} not found.")
        return

    sample_df = pd.read_csv(sample_path)
    my_df = pd.read_csv(my_path)
    
    print(f"Sample shape: {sample_df.shape}")
    print(f"My Submission shape: {my_df.shape}")
    
    my_df['file_id'] = my_df['filename'].apply(lambda x: os.path.splitext(x)[0])
    my_df = my_df.groupby('file_id')['prob'].mean().reset_index()
    
    sample_df['file_id'] = sample_df['filename'].apply(lambda x: os.path.splitext(x)[0])
    merged_df = pd.merge(sample_df[['filename', 'file_id']], my_df, on='file_id', how='left')
    
    nans = merged_df['prob'].isna().sum()
    if nans > 0:
        print(f"Warning: {nans} files have no predictions. Filling with 0.5.")
        merged_df['prob'] = merged_df['prob'].fillna(0.5)
        
    final_df = merged_df[['filename', 'prob']]
    final_df.to_csv(output_path, index=False)
    print(f"Fixed submission saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=str, default='data/submission/sample_submission.csv')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    fix_submission(args.sample, args.input, args.output)