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
            
            # 3. Label Assignment (Strict)
            for f in all_files:
                parts = f.replace('\\', '/').split('/')
                parent = parts[-2].lower()
                grandparent = parts[-3].lower() if len(parts) > 2 else ""
                
                label = None
                
                # Check immediate parent or grandparent for keywords
                # FF++ / 140k structure: .../real/0001.jpg OR .../real-vs-fake/real/0001.jpg
                if parent == 'real' or parent == '0':
                    label = 0
                elif parent == 'fake' or parent == '1':
                    label = 1
                
                # Celeb-DF structure: .../Celeb-real/id0/video.mp4 (frames)
                elif 'cele' in parent and 'real' in parent:
                    label = 0
                elif 'cele' in parent and 'synthesis' in parent: # Celeb-synthesis
                    label = 1
                elif 'youtube' in parent and 'real' in parent: # YouTube-real
                    label = 0
                
                # Grandparent check (if images are in video folders)
                elif parent.startswith('id') or parent.isdigit(): # .../Celeb-real/id0/frame.jpg
                    if 'real' in grandparent and 'fake' not in grandparent:
                        label = 0
                    elif 'fake' in grandparent or 'synthesis' in grandparent:
                        label = 1
                    elif 'youtube' in grandparent:
                        label = 0
                
                if label is None:
                    # Fallback: Look for specific keywords in the full path relative to root
                    rel_path = os.path.relpath(f, root).lower().replace('\\', '/')
                    if '/real/' in rel_path or '/0/' in rel_path:
                        label = 0
                    elif '/fake/' in rel_path or '/1/' in rel_path:
                        label = 1
                
                if label is not None:
                    self.image_paths.append(f)
                    self.labels.append(label)
                    if label == 0: real_count += 1
                    else: fake_count += 1
            
        print(f"[{mode.upper()}] Total Loaded {len(self.image_paths)} images (Real: {real_count}, Fake: {fake_count})")

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = cv2.imread(self.image_paths[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform: img = self.transform(image=img)['image']
            return img, self.labels[idx]
        except: return torch.zeros((224,224,3), dtype=torch.uint8).numpy(), 0
