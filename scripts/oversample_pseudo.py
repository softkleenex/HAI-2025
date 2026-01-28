import pandas as pd
import shutil
import os

# 1. Load original pseudo labels
input_path = 'data/pseudo_train.csv'
output_path = 'data/pseudo_train_10x.csv'

if not os.path.exists(input_path):
    print(f"Error: {input_path} not found. Run create_pseudo_labels.py first.")
    exit()

df = pd.read_csv(input_path)
print(f"Original Samples: {len(df)}")

# 2. Oversample (x10)
# Simply concatenate the dataframe 10 times
dfs = [df] * 10
df_10x = pd.concat(dfs, ignore_index=True)

print(f"Oversampled Samples: {len(df_10x)}")

# 3. Save
df_10x.to_csv(output_path, index=False)
print(f"Saved to {output_path}")

# 4. Swap files (Backup original)
shutil.copy(input_path, 'data/pseudo_train_backup.csv')
shutil.copy(output_path, 'data/pseudo_train.csv')
print("✅ Swapped 'data/pseudo_train.csv' with 10x version.")
