import pandas as pd
import argparse
import os

def ensemble_submissions(input_files, weights, output_path, sample_path):
    if len(input_files) != len(weights):
        print("Error: Number of input files must match number of weights.")
        return

    # Normalize weights
    weights = [float(w) for w in weights]
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    print(f"Ensembling {len(input_files)} files with weights: {weights}")

    # Load sample to preserve order and file_id mapping
    if not os.path.exists(sample_path):
        print(f"Error: Sample submission not found at {sample_path}")
        return
        
    sample = pd.read_csv(sample_path)
    sample['file_id'] = sample['filename'].apply(lambda x: os.path.splitext(x)[0])
    
    final_prob = 0.0
    
    for i, file_path in enumerate(input_files):
        print(f"Loading {file_path} (Weight: {weights[i]:.2f})...")
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found.")
            return

        df = pd.read_csv(file_path)
        
        # Robust alignment: matching by file_id (ignoring extension differences)
        df['file_id'] = df['filename'].apply(lambda x: os.path.splitext(x)[0])
        
        # Merge left on sample to ensure we have all required rows in correct order
        # We merge sample[['file_id']] with df[['file_id', 'prob']]
        merged = pd.merge(sample[['file_id']], df[['file_id', 'prob']], on='file_id', how='left')
        
        # Handle missing values
        nans = merged['prob'].isna().sum()
        if nans > 0:
            print(f"Warning: {nans} missing predictions in {file_path}. Filling with 0.5.")
        
        probs = merged['prob'].fillna(0.5).values
        final_prob += probs * weights[i]

    submission = sample[['filename']].copy()
    submission['prob'] = final_prob
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False, float_format='%.6f')
    print(f"Ensemble saved to {output_path}")
    print(submission.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble multiple submission CSVs.")
    parser.add_argument('--inputs', nargs='+', required=True, help='List of input CSV paths (space separated)')
    parser.add_argument('--weights', nargs='+', required=True, type=float, help='List of weights corresponding to inputs (space separated)')
    parser.add_argument('--output', type=str, required=True, help='Output CSV path')
    parser.add_argument('--sample', type=str, default='data/submission/sample_submission.csv', help='Path to sample submission')
    
    args = parser.parse_args()
    ensemble_submissions(args.inputs, args.weights, args.output, args.sample)