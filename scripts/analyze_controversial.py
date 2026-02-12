import cv2
import os
import glob
import numpy as np

# 1. 파일 경로 매핑
processed_dir = 'data/processed/test'
target_files = ['TEST_015.mp4', 'TEST_106.mp4', 'TEST_176.mp4', 'TEST_244.mp4', 'TEST_318.mp4']

print(f"🕵️‍♂️ [Deep Dive Analysis] Analyzing {len(target_files)} controversial files...\n")

for filename in target_files:
    base = os.path.splitext(filename)[0]
    folder = os.path.join(processed_dir, base)
    
    image_path = None
    if os.path.isdir(folder):
        imgs = glob.glob(os.path.join(folder, '*.jpg'))
        if imgs:
            image_path = imgs[0]
    
    print(f"📁 File: {filename}")
    if image_path:
        # 이미지 로드
        img = cv2.imread(image_path)
        
        if img is None:
            print("   ❌ Failed to load image.")
            continue
            
        h, w, c = img.shape
        mean_brightness = np.mean(img)
        std_brightness = np.std(img)
        
        # Laplacian Variance (Blur 정도 측정 - 낮을수록 흐릿함)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        print(f"   ✅ Image Found: {os.path.basename(image_path)}")
        print(f"   📏 Resolution: {w}x{h}")
        print(f"   💡 Brightness: {mean_brightness:.1f} (Mean)")
        print(f"   🌫️ Sharpness (Laplacian Var): {blur_score:.1f} (Lower = Blurry)")
        
        if blur_score < 100:
            print("   ⚠️  Warning: Image seems BLURRY. (May confuse the model)")
        if mean_brightness < 40:
             print("   ⚠️  Warning: Image seems DARK. (Hard to detect details)")
             
    else:
        print("   ❌ No processed image found (maybe raw video wasn't processed?)")
    print("-" * 40)
