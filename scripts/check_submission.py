import pandas as pd

sample = pd.read_csv('data/submission/sample_submission.csv')
target = pd.read_csv('data/submission/final_ensemble_v4.csv')

print("Shape:", sample.shape, target.shape)

# Check columns
print("Columns:", sample.columns.tolist(), target.columns.tolist())

# Check filenames equality
if not sample['filename'].equals(target['filename']):
    print("FATAL: Filenames do not match!")
    # Find diff
    diff = sample['filename'] != target['filename']
    print(sample[diff].head())
    print(target[diff].head())
else:
    print("Filenames match exactly.")

# Check probabilities
print("Prob stats:")
print(target['prob'].describe())

# Check for NaN/Inf
import numpy as np
if np.isinf(target['prob']).any():
    print("FATAL: Infinity found in prob!")
else:
    print("No Inf found.")

if target['prob'].isnull().any():
    print("FATAL: NaN found in prob!")
else:
    print("No NaN found.")
