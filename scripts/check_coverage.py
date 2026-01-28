import pandas as pd
import os

sample = pd.read_csv('data/submission/sample_submission.csv')
dino = pd.read_csv('data/submission/submission_dino_large_best_mean.csv')

sample['file_id'] = sample['filename'].apply(lambda x: os.path.splitext(x)[0])
dino['file_id'] = dino['filename'].apply(lambda x: os.path.splitext(x)[0])

missing = set(sample['file_id']) - set(dino['file_id'])
print(f"Missing in DINO: {len(missing)}")
if len(missing) > 0:
    print(missing)

# Check extra
extra = set(dino['file_id']) - set(sample['file_id'])
print(f"Extra in DINO: {len(extra)}")
print(extra)
