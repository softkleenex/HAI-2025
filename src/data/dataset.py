import os
import cv2
import glob
import torch
from torch.utils.data import Dataset

class DeepFakeDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
        # 1. Determine Target Directory
        # Try to find specific split folder
        potential_dirs = [
            os.path.join(root_dir, mode), 
            os.path.join(root_dir, 'real-vs-fake', mode),
            os.path.join(root_dir, 'real_vs_fake', mode),
            root_dir # Fallback to root
        ]
        
        target_dir = root_dir
        for p in potential_dirs:
            if os.path.exists(p) and os.path.isdir(p):
                # Check if it contains images or subfolders
                if glob.glob(os.path.join(p, '**', '*.jpg'), recursive=True):
                    target_dir = p
                    break
        
        print(f"[{mode.upper()}] Scanning for images in: {target_dir}")
        
        # 2. Recursive Search for Images
        all_files = glob.glob(os.path.join(target_dir, '**', '*.jpg'), recursive=True)
        all_files += glob.glob(os.path.join(target_dir, '**', '*.png'), recursive=True)
        all_files += glob.glob(os.path.join(target_dir, '**', '*.jpeg'), recursive=True)
        
        self.image_paths = []
        self.labels = []
        
        real_count = 0
        fake_count = 0
        
        # 3. Label Assignment based on Path
        for f in all_files:
            # Check keywords in path (robust)
            lower_path = f.lower().replace('\\', '/')
            if 'real' in lower_path and 'fake' not in lower_path.split('/')[-1]: 
                # Avoid 'real_vs_fake' folder name matching, check strictly
                label = 0
                real_count += 1
            elif 'fake' in lower_path:
                label = 1
                fake_count += 1
            else:
                continue # Skip unclear files
            
            self.image_paths.append(f)
            self.labels.append(label)
            
        print(f"[{mode.upper()}] Loaded {len(self.image_paths)} images (Real: {real_count}, Fake: {fake_count})")

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = cv2.imread(self.image_paths[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform: img = self.transform(image=img)['image']
            return img, self.labels[idx]
        except: return torch.zeros((224,224,3), dtype=torch.uint8).numpy(), 0
