import os
import cv2
import glob
import torch
from torch.utils.data import Dataset

class DeepFakeDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        # Support multiple root directories
        if isinstance(root_dir, str):
            self.root_dirs = [root_dir]
        else:
            self.root_dirs = root_dir
            
        self.transform = transform
        self.mode = mode
        
        self.image_paths = []
        self.labels = []
        
        real_count = 0
        fake_count = 0
        
        for root in self.root_dirs:
            print(f"[{mode.upper()}] Scanning root: {root}")
            # 1. Determine Target Directory
            potential_dirs = [
                os.path.join(root, mode), 
                os.path.join(root, 'real-vs-fake', mode),
                os.path.join(root, 'real_vs_fake', mode),
                root # Fallback to root
            ]
            
            target_dir = root
            for p in potential_dirs:
                if os.path.exists(p) and os.path.isdir(p):
                    if glob.glob(os.path.join(p, '**', '*.jpg'), recursive=True):
                        target_dir = p
                        break
            
            # 2. Recursive Search
            all_files = glob.glob(os.path.join(target_dir, '**', '*.jpg'), recursive=True)
            all_files += glob.glob(os.path.join(target_dir, '**', '*.png'), recursive=True)
            
            # 3. Label Assignment
            for f in all_files:
                lower_path = f.lower().replace('\\', '/')
                # Robust label logic
                if 'real' in lower_path and 'fake' not in lower_path.split('/')[-1]: 
                    label = 0
                    real_count += 1
                elif 'fake' in lower_path:
                    label = 1
                    fake_count += 1
                elif 'youtube' in lower_path: # Celeb-DF real structure often uses 'YouTube-real'
                    label = 0
                    real_count += 1
                elif 'celeb' in lower_path and 'real' in lower_path: # Celeb-real
                    label = 0
                    real_count += 1
                elif 'celeb' in lower_path and 'synthesis' in lower_path: # Celeb-synthesis (fake)
                    label = 1
                    fake_count += 1
                else:
                    continue 
                
                self.image_paths.append(f)
                self.labels.append(label)
            
        print(f"[{mode.upper()}] Total Loaded {len(self.image_paths)} images (Real: {real_count}, Fake: {fake_count})")

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = cv2.imread(self.image_paths[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform: img = self.transform(image=img)['image']
            return img, self.labels[idx]
        except: return torch.zeros((224,224,3), dtype=torch.uint8).numpy(), 0
