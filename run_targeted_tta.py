import os
import sys
# 프로젝트 루트 경로를 sys.path에 추가
sys.path.append(os.getcwd())

import glob
import yaml
import torch
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn.functional as F

# 모듈 로드
from src.models.dino import DeepFakeDINOv2
from src.models.model import DeepFakeClassifier # EfficientNet용 클래스 추가

# 🎯 검증할 파일 목록
TARGET_FILES = ['TEST_015.mp4', 'TEST_106.mp4', 'TEST_176.mp4', 'TEST_244.mp4', 'TEST_318.mp4']
DATA_DIR = 'data/processed/test'

def find_best_model():
    # 1. checkpoints 폴더 내 모든 pth, pt 파일 검색 (하위 폴더 포함)
    ckpt_dir = 'checkpoints'
    best_models = glob.glob(os.path.join(ckpt_dir, '**', '*.pth'), recursive=True)
    best_models += glob.glob(os.path.join(ckpt_dir, '**', '*.pt'), recursive=True)
    
    print(f"🔍 Searching in {os.path.abspath(ckpt_dir)}...")
    if best_models:
        print(f"✅ Found {len(best_models)} model files.")
    else:
        # 디버깅: 폴더 내용 직접 출력
        try:
            print(f"📁 Directory listing of {ckpt_dir}: {os.listdir(ckpt_dir)}")
        except Exception as e:
            print(f"❌ Error listing directory: {e}")

    if not best_models:
        print("❌ No checkpoint found in 'checkpoints/'!")
        return None, None

    # 'best'가 포함된 모델 우선, 없으면 가장 최근 모델
    best_candidates = [m for m in best_models if 'best' in m.lower()]
    target_models = best_candidates if best_candidates else best_models
    
    latest_ckpt = max(target_models, key=os.path.getmtime)
    print(f"📂 Using Checkpoint: {latest_ckpt}")
    
    # Config 및 모델 타입 추론
    model_type = 'dino' # Default
    if 'dino' in latest_ckpt.lower():
        model_type = 'dino'
        if 'large' in latest_ckpt.lower():
            config_path = 'configs/dino_large.yaml'
        else:
            config_path = 'configs/dino_base_finetune.yaml'
    elif 'efficientnet' in latest_ckpt.lower() or 'effnet' in latest_ckpt.lower():
        model_type = 'effnet'
        config_path = 'configs/effnet_b5_recursive.yaml' # recursive 모델이 발견되었으므로 변경
    else:
        # Fallback
        config_path = 'configs/dino_large.yaml' 
        
    print(f"⚙️  Using Config: {config_path} (Type: {model_type})")
    return latest_ckpt, config_path, model_type

def run_targeted_tta():
    # model_type도 반환받음
    find_result = find_best_model()
    if not find_result or find_result[0] is None: return
    ckpt_path, config_path, model_type = find_result

    with open(config_path, 'r') as f: config = yaml.safe_load(f)
    img_size = config['data']['img_size']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Model
    print("⏳ Loading Model...")
    try:
        if model_type == 'dino':
            model = DeepFakeDINOv2(config).to(device)
        else:
            # EfficientNet 등 일반 모델 (DeepFakeClassifier는 config dict가 아니라 개별 인자를 받음)
            model_cfg = config['model']
            model = DeepFakeClassifier(
                backbone=model_cfg['backbone'],
                pretrained=model_cfg['pretrained'],
                num_classes=model_cfg['num_classes'],
                dropout=model_cfg['dropout']
            ).to(device)
            print(f"✅ Loaded DeepFakeClassifier with {model_cfg['backbone']}")
    except Exception as e:
        print(f"⚠️  Model load failed ({e}). Check config match.")
        sys.exit(1)
        
    state_dict = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
    
    # 키 불일치(module. prefix 등) 처리
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # Define TTA Transforms
    transforms = {
        'Original': A.Compose([A.Resize(img_size, img_size), A.Normalize(), ToTensorV2()]),
        'Flip': A.Compose([A.Resize(img_size, img_size), A.HorizontalFlip(p=1.0), A.Normalize(), ToTensorV2()]),
        'Sharpen': A.Compose([A.Resize(img_size, img_size), A.Sharpen(p=1.0), A.Normalize(), ToTensorV2()]),
        'Blur': A.Compose([A.Resize(img_size, img_size), A.GaussianBlur(p=1.0), A.Normalize(), ToTensorV2()]),
    }

    results = []

    print("\n🚀 Starting TTA Analysis for 5 Controversial Files...\n")
    
    for vid_name in TARGET_FILES:
        base_name = os.path.splitext(vid_name)[0]
        folder_path = os.path.join(DATA_DIR, base_name)
        
        img_paths = glob.glob(os.path.join(folder_path, '*.jpg'))
        if not img_paths:
            print(f"⚠️  No images found for {vid_name}")
            continue
            
        img_path = img_paths[0]
        original_img = cv2.imread(img_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        print(f"🎬 Processing: {vid_name}")
        
        vid_probs = {}
        for tta_name, transform in transforms.items():
            aug = transform(image=original_img)['image'].unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(aug)
                # Output shape: [1, 2] -> Softmax -> get class 1 (FAKE) probability
                probs = torch.softmax(output, dim=1)
                prob = probs[0, 1].item()
            vid_probs[tta_name] = prob
            
        avg_prob = np.mean(list(vid_probs.values()))
        final_verdict = "FAKE" if avg_prob > 0.5 else "REAL"
        
        print(f"   🔹 Original: {vid_probs['Original']:.4f}")
        print(f"   🔹 Flip:     {vid_probs['Flip']:.4f}")
        print(f"   🔹 Sharpen:  {vid_probs['Sharpen']:.4f}  <-- Check this!")
        print(f"   🔹 Blur:     {vid_probs['Blur']:.4f}")
        print(f"   👉 Average:  {avg_prob:.4f} ({final_verdict})")
        
        if abs(vid_probs['Sharpen'] - vid_probs['Blur']) > 0.3:
            print("   ⚠️  High Sensitivity to Image Quality! (Blur changes prediction)")
            
        print("-" * 30)
        
        results.append({
            'filename': vid_name,
            **vid_probs,
            'Average': avg_prob
        })

    df = pd.DataFrame(results)
    print("\n📊 Summary Table:")
    print(df)

if __name__ == "__main__":
    run_targeted_tta()
