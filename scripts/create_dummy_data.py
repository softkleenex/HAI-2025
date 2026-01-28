import os
import cv2
import numpy as np
from pathlib import Path

def create_dummy_dataset(root_dir):
    root = Path(root_dir)
    
    # Structure for 140k dataset
    subsets = ['train', 'valid', 'test']
    classes = ['real', 'fake']
    
    for subset in subsets:
        for cls in classes:
            dir_path = root / 'real_vs_fake' / 'real-vs-fake' / subset / cls
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create 5 dummy images per class per subset
            for i in range(5):
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                cv2.imwrite(str(dir_path / f"dummy_{i}.jpg"), img)
                
    print(f"Created dummy dataset at {root}")

if __name__ == "__main__":
    create_dummy_dataset("data/raw")
