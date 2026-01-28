import os
import cv2
import glob
import argparse
import torch
from tqdm import tqdm
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np

def preprocess_videos(input_dir, output_dir, device='cuda', sample_interval=10):
    """
    Extract frames and crop faces from videos AND images.
    Args:
        input_dir: Directory containing mp4/mov/jpg/png files.
        output_dir: Directory to save cropped face images.
        device: 'cuda' or 'cpu'
        sample_interval: Extract 1 frame every N frames (for videos).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize MTCNN
    mtcnn = MTCNN(keep_all=False, select_largest=True, device=device, margin=20, post_process=False)

    # Collect all files
    video_exts = ['*.mp4', '*.mov', '*.avi']
    image_exts = ['*.jpg', '*.jpeg', '*.png', '*.jfif']
    
    files = []
    for ext in video_exts + image_exts:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
        # Case insensitive search might be needed, but glob usually works if extensions are lower case.
        # Let's add upper case just in case.
        files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
        
    files = sorted(list(set(files))) # Remove duplicates
    print(f"Found {len(files)} files in {input_dir}")

    for file_path in tqdm(files, desc="Processing Files"):
        filename = os.path.basename(file_path)
        file_id = os.path.splitext(filename)[0]
        file_output_dir = os.path.join(output_dir, file_id)
        
        # Skip if already processed (Optional: remove this check if re-run needed)
        # if os.path.exists(file_output_dir) and len(os.listdir(file_output_dir)) > 0:
        #     continue
            
        os.makedirs(file_output_dir, exist_ok=True)
        
        ext = os.path.splitext(filename)[1].lower()
        
        frames = []
        
        if ext in ['.mp4', '.mov', '.avi']:
            # Process Video
            cap = cv2.VideoCapture(file_path)
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % sample_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))
                frame_count += 1
            cap.release()
            
        else:
            # Process Image
            try:
                img = cv2.imread(file_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(img_rgb))
            except Exception as e:
                print(f"Error reading image {file_path}: {e}")

        # Batch detection usually better, but for simplicity let's loop
        # Or stack frames if memory allows. 
        # For simplicity and robustness: process each frame.
        
        for i, frame in enumerate(frames):
            try:
                # Detect and save
                save_path = os.path.join(file_output_dir, f"frame_{i}.jpg")
                mtcnn(frame, save_path=save_path)
            except Exception as e:
                pass 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/test', help='Input directory')
    parser.add_argument('--output_dir', type=str, default='data/processed/test', help='Output directory')
    parser.add_argument('--sample_interval', type=int, default=10, help='Frame sampling interval for videos')
    args = parser.parse_args()
    
    preprocess_videos(args.input_dir, args.output_dir, sample_interval=args.sample_interval)