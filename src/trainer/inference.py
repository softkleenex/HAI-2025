import os
import glob
import yaml
import argparse
import torch
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn.functional as F

from src.models.model import DeepFakeClassifier
from src.models.vit_baseline import DeepFakeViT
from src.models.dino import DeepFakeDINOv2

class InferenceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.video_ids = []
        
        video_folders = glob.glob(os.path.join(root_dir, '*'))
        for folder in video_folders:
            if not os.path.isdir(folder): continue
            video_id = os.path.basename(folder)
            files = glob.glob(os.path.join(folder, '*.jpg'))
            for f in files:
                self.image_paths.append(f)
                self.video_ids.append(video_id)
        print(f"[INFERENCE] Found {len(self.image_paths)} frames from {len(video_folders)} videos.")

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = cv2.imread(img_path)
            if image is None: raise ValueError
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            
        if self.transform:
            image = self.transform(image=image)['image']
        return image, self.video_ids[idx]

def inference(config_path, checkpoint_path, output_csv, aggregation='mean', tta=False):
    with open(config_path, 'r') as f: config = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_size = config['data']['img_size']

    # Base Transform
    base_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    # TTA Transforms (List of compositions)
    tta_transforms = []
    if tta:
        print("🔥 Test Time Augmentation (TTA) Enabled!")
        # 1. Original
        tta_transforms.append(base_transform)
        # 2. Horizontal Flip
        tta_transforms.append(A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]))
        # 3. Vertical Flip (Video often has vertical artifacts?) - Maybe not for face.
        # Let's try slight Zoom/Crop instead.
        tta_transforms.append(A.Compose([
            A.Resize(int(img_size*1.1), int(img_size*1.1)),
            A.CenterCrop(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]))
    else:
        tta_transforms.append(base_transform)

    # We need a custom dataset or loader loop to handle TTA
    # Simple approach: Load image, apply N transforms, stack batch, predict, mean.
    
    test_dataset = InferenceDataset('data/processed/test', transform=None) # No transform here
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4) # Batch 1 for TTA logic simplicity

    # Model Selection
    backbone = config['model']['backbone']
    if backbone == 'vit_baseline':
        model = DeepFakeViT(num_classes=config['model']['num_classes']).to(device)
    elif 'dino_v2' in backbone:
        model_name = config['model'].get('model_name', 'facebook/dinov2-base')
        model = DeepFakeDINOv2(model_name=model_name, num_classes=config['model']['num_classes']).to(device)
    else:
        model = DeepFakeClassifier(backbone=backbone, pretrained=False, num_classes=config['model']['num_classes']).to(device)

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")

    model.eval()
    video_preds = {}
    
    with torch.no_grad():
        for image_np, video_ids in tqdm(test_loader, desc="Inference"):
            # image_np is batch of 1 image (H, W, C) from dataset (before transform)
            image_np = image_np[0].numpy()
            
            tta_probs = []
            for t in tta_transforms:
                aug_img = t(image=image_np)['image'].unsqueeze(0).to(device)
                outputs = model(aug_img)
                probs = F.softmax(outputs, dim=1)[:, 1].cpu().item()
                tta_probs.append(probs)
            
            # Mean of TTA predictions
            final_prob = np.mean(tta_probs)
            
            vid = video_ids[0]
            if vid not in video_preds: video_preds[vid] = []
            video_preds[vid].append(final_prob)

    results = []
    for vid, probs in video_preds.items():
        if aggregation == 'mean':
            prob = np.mean(probs)
        elif aggregation == 'max':
            prob = np.max(probs)
        elif aggregation == 'topk':
            k = max(1, int(len(probs) * 0.2))
            prob = np.mean(np.sort(probs)[-k:])
        else:
            prob = np.mean(probs)
        results.append({'filename': vid + '.mp4', 'prob': prob})

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved submission to {output_csv} (Aggregation: {aggregation}, TTA: {tta})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base_config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--aggregation', type=str, default='mean', choices=['mean', 'max', 'topk'], help='Aggregation method')
    parser.add_argument('--tta', action='store_true', help='Enable Test Time Augmentation')
    args = parser.parse_args()
    inference(args.config, args.checkpoint, args.output, args.aggregation, args.tta)