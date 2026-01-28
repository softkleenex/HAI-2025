import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import cv2

from src.models.model import DeepFakeClassifier
from src.models.vit_baseline import DeepFakeViT
from src.models.dino import DeepFakeDINOv2 # Import DINO
from src.data.dataset import DeepFakeDataset

class PseudoLabelDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        print(f"Loaded PseudoLabelDataset with {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['path']
        label = int(self.data.iloc[idx]['label'])
        
        try:
            image = cv2.imread(img_path)
            if image is None: raise ValueError
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            # Fallback for errors
            print(f"Error loading {img_path}: {e}")
            image = torch.zeros((224, 224, 3), dtype=torch.uint8).numpy()

        if self.transform:
            image = self.transform(image=image)['image']
            
        return image, label

def train(config_path, resume_path=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['train']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    img_size = config['data']['img_size']

    # --- Light Augmentation for Linear Probing ---
    # Since we freeze backbone, strong aug might be too hard for just a linear layer.
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.CoarseDropout(max_holes=4, max_height=img_size//10, max_width=img_size//10, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # ImageNet stats for DINO
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # Dataset
    train_dataset = DeepFakeDataset(config['data']['train_dir'], transform=train_transform, mode='train')
    
    # --- Pseudo Labeling Injection ---
    pseudo_csv = 'data/pseudo_train.csv'
    if os.path.exists(pseudo_csv):
        print(f"Adding Pseudo-Label Dataset from {pseudo_csv}")
        pseudo_dataset = PseudoLabelDataset(pseudo_csv, transform=train_transform)
        train_dataset = ConcatDataset([train_dataset, pseudo_dataset])
    else:
        print("No pseudo-label data found.")
        
    val_dataset = DeepFakeDataset(config['data']['val_dir'], transform=val_transform, mode='val')

    print(f"[TRAIN] Total images: {len(train_dataset)}")
    print(f"[VAL] Total images: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], 
                              shuffle=True, num_workers=config['data']['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], 
                            shuffle=False, num_workers=config['data']['num_workers'], pin_memory=True)

    # Model Selection
    backbone = config['model']['backbone']
    if backbone == 'vit_baseline':
        model = DeepFakeViT(num_classes=config['model']['num_classes']).to(device)
    elif 'dino_v2' in backbone: # Handles dino_v2, dino_v2_large
        model_name = config['model'].get('model_name', 'facebook/dinov2-base')
        freeze = config['model'].get('freeze_backbone', True)
        model = DeepFakeDINOv2(model_name=model_name, num_classes=config['model']['num_classes'], freeze_backbone=freeze).to(device)
    else:
        model = DeepFakeClassifier(
            backbone=backbone,
            pretrained=config['model']['pretrained'],
            num_classes=config['model']['num_classes'],
            dropout=config['model']['dropout']
        ).to(device)

    # Optimizer (Only optimize parameters that require grad)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=config['train']['lr'], 
                            weight_decay=float(config['train']['weight_decay']))
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['train']['epochs'], eta_min=float(config['train']['min_lr']))
    scaler = GradScaler()
    
    accumulation_steps = config['train'].get('gradient_accumulation_steps', 1)
    print(f"Gradient Accumulation Steps: {accumulation_steps}")

    start_epoch = 0
    best_acc = 0.0

    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'epoch' in checkpoint: start_epoch = checkpoint['epoch'] + 1
        if 'best_acc' in checkpoint: best_acc = checkpoint['best_acc']
        print(f"Resumed at Epoch {start_epoch}")

    criterion = nn.CrossEntropyLoss()
    os.makedirs(config['train']['save_dir'], exist_ok=True)

    for epoch in range(start_epoch, config['train']['epochs']):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        optimizer.zero_grad() # Initialize gradients once at start of epoch
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}")
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps # Normalize loss
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps # Scale back for logging
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': train_loss / (total/config['data']['batch_size']), 'acc': 100 * correct / total})

        scheduler.step()
        
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        print(f"Validation Accuracy: {val_acc:.2f}%")

        # Save Checkpoint (Best & Snapshot)
        state = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'epoch': epoch
        }

        if val_acc > best_acc:
            best_acc = val_acc
            state['best_acc'] = best_acc # Update best_acc in state
            save_path = os.path.join(config['train']['save_dir'], f"{config['experiment_name']}_best.pt")
            torch.save(state, save_path)
            print(f"Saved Best Model to {save_path}")
        
        # Always save snapshot at end of epoch
        snapshot_path = os.path.join(config['train']['save_dir'], f"{config['experiment_name']}_epoch{epoch+1}_snapshot.pt")
        torch.save(state, snapshot_path)
        print(f"Saved Snapshot to {snapshot_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base_config.yaml')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    train(args.config, args.resume)
