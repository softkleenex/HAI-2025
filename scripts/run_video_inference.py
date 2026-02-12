import torch
import torch.nn as nn
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import glob
import os
import cv2
import numpy as np
from tqdm import tqdm

# Define Model Architecture (Must match training)
class CNNLSTM(nn.Module):
    def __init__(self, seq_length=10, hidden_dim=256, num_classes=2):
        super(CNNLSTM, self).__init__()
        self.seq_length = seq_length
        resnet = models.resnet18(pretrained=False) # Weights loaded later
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn_out_dim = 512
        self.lstm = nn.LSTM(input_size=self.cnn_out_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        b, s, c, h, w = x.size()
        c_in = x.view(b * s, c, h, w)
        features = self.cnn(c_in)
        features = features.view(b, s, -1)
        lstm_out, _ = self.lstm(features)
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        return out

def inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Model
    model = CNNLSTM().to(device)
    ckpt_path = 'checkpoints/video_model_best.pt'
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: {ckpt_path} not found!")
        return

    print("Loading checkpoint...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # Transform
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ])
    
    # Find Test Videos (Folders)
    test_root = 'data/processed/test'
    video_folders = glob.glob(os.path.join(test_root, '*'))
    print(f"Found {len(video_folders)} test videos.")
    
    results = []
    
    for folder in tqdm(video_folders):
        if not os.path.isdir(folder): continue
        video_id = os.path.basename(folder) # TEST_000
        
        # Load Frames
        frames_paths = sorted(glob.glob(os.path.join(folder, '*.jpg')))
        
        if len(frames_paths) == 0:
            print(f"⚠️ Warning: No frames for {video_id}")
            results.append({'filename': f"{video_id}.mp4", 'prob': 0.5})
            continue
            
        # Select 10 frames (uniform or random)
        # Use first 10, or pad if less
        selected_paths = frames_paths[:10]
        
        frames = []
        for p in selected_paths:
            img = cv2.imread(p)
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transform(image=img)['image']
            frames.append(img)
            
        if len(frames) == 0:
            results.append({'filename': f"{video_id}.mp4", 'prob': 0.5})
            continue
            
        # Pad loop
        while len(frames) < 10:
            frames.append(frames[-1])
            
        frames_tensor = torch.stack(frames).unsqueeze(0).to(device) # [1, 10, C, H, W]
        
        with torch.no_grad():
            out = model(frames_tensor)
            prob = torch.softmax(out, dim=1)[0, 1].item()
            
        results.append({'filename': f"{video_id}.mp4", 'prob': prob})
        
    # Save
    df = pd.DataFrame(results)
    output_path = 'data/submission/submission_video_fixed.csv'
    df.to_csv(output_path, index=False)
    print(f"✅ Saved video inference to {output_path}")

if __name__ == '__main__':
    inference()
