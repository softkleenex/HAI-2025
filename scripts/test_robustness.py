import os
import yaml
import torch
import random
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score

from src.models.model import DeepFakeClassifier
from src.models.vit_baseline import DeepFakeViT
from src.data.dataset import DeepFakeDataset

def test_robustness(config_path, model_checkpoint="none"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    img_size = config['data']['img_size']

    # --- Robustness Scenarios ---
    scenarios = {
        "Clean": A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]),
        "Compression (Low Quality)": A.Compose([
            A.Resize(img_size, img_size),
            A.ImageCompression(quality_lower=40, quality_upper=50, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]),
        "Gaussian Blur": A.Compose([
            A.Resize(img_size, img_size),
            A.GaussianBlur(blur_limit=(7, 11), p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]),
        "Gaussian Noise": A.Compose([
            A.Resize(img_size, img_size),
            A.GaussNoise(var_limit=(50.0, 100.0), p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    }

    # Load Model
    backbone = config['model']['backbone']
    if backbone == 'vit_baseline':
        model = DeepFakeViT(num_classes=config['model']['num_classes']).to(device)
    else:
        model = DeepFakeClassifier(
            backbone=backbone,
            pretrained=False,
            num_classes=config['model']['num_classes']
        ).to(device)
        
        if model_checkpoint != "none" and os.path.exists(model_checkpoint):
            model.load_state_dict(torch.load(model_checkpoint, map_location=device))
            print(f"Loaded checkpoint: {model_checkpoint}")
        else:
            print("Warning: No checkpoint loaded for custom model!")

    model.eval()

    # Run Test per Scenario
    print("\n--- Robustness Test Results ---")
    print(f"Model: {backbone}")
    
    val_dir = config['data']['val_dir']
    
    for name, transform in scenarios.items():
        dataset = DeepFakeDataset(val_dir, transform=transform, mode='val')
        
        # Shuffle and Limit
        if len(dataset.image_paths) > 500:
            combined = list(zip(dataset.image_paths, dataset.labels))
            random.seed(42) # Fixed seed for reproducibility
            random.shuffle(combined)
            dataset.image_paths, dataset.labels = zip(*combined[:500])
            dataset.image_paths = list(dataset.image_paths)
            dataset.labels = list(dataset.labels)
            
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(loader, desc=f"Testing {name}", leave=False):
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1] # Fake prob
                
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Metrics
        if len(set(all_labels)) < 2:
            auc = 0.5
            print("Warning: Only one class present in subset.")
        else:
            auc = roc_auc_score(all_labels, all_preds)
            
        acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
        
        print(f"[{name}] AUC: {auc:.4f} | Acc: {acc:.4f}")

if __name__ == '__main__':
    # Default to config
    test_robustness("configs/base_config.yaml", "none")
