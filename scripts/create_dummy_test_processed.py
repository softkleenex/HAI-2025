import os
import cv2
import numpy as np
from pathlib import Path

def create_dummy_processed_test(root_dir):
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    
    # Create 5 dummy video folders
    for i in range(5):
        video_id = f"test_video_{i:03d}"
        video_dir = root / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # Create 10 frames per video
        for j in range(10):
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            cv2.imwrite(str(video_dir / f"frame_{j}.jpg"), img)
                
    print(f"Created dummy processed test data at {root}")

if __name__ == "__main__":
    create_dummy_processed_test("data/processed/test")
