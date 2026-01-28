import os
import cv2
import glob
import numpy as np
from torch.utils.data import Dataset, ConcatDataset

class SingleDirDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        self.class_map = {'real': 0, 'fake': 1}
        
        for label_name, label_idx in self.class_map.items():
            class_dir = os.path.join(root_dir, label_name)
            if not os.path.exists(class_dir): continue
            
            # Recursive glob
            files = glob.glob(os.path.join(class_dir, '**', '*.*'), recursive=True)
            # Filter extensions
            files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif'))]
            
            self.image_paths.extend(files)
            self.labels.extend([label_idx] * len(files))
            
    def __len__(self): return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = cv2.imread(img_path)
            if image is None: raise ValueError
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            return self.__getitem__((idx + 1) % len(self))
            
        if self.transform:
            image = self.transform(image=image)['image']
        return image, label

class DeepFakeDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        """
        Loads from FF++ and optionally Pseudo-Label dir if mode is train.
        """
        self.main_dataset = SingleDirDataset(root_dir, transform)
        self.datasets = [self.main_dataset]
        
        # Add Pseudo-Labels if training
        if mode == 'train':
            pseudo_dir = "data/raw/pseudo_train"
            if os.path.exists(pseudo_dir):
                print(f"Adding Pseudo-Label Dataset from {pseudo_dir}")
                self.pseudo_dataset = SingleDirDataset(pseudo_dir, transform)
                self.datasets.append(self.pseudo_dataset)
        
        self.concat_dataset = ConcatDataset(self.datasets)
        print(f"[{mode.upper()}] Total images: {len(self.concat_dataset)}")

    def __len__(self): return len(self.concat_dataset)
    def __getitem__(self, idx): return self.concat_dataset[idx]
