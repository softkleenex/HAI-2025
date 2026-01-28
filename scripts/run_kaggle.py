import os
import glob

def run_kaggle():
    # 1. Install Libraries (Silently)
    print("📦 Installing libraries (Silently)...")
    # Redirect stdout/stderr to devnull to hide progress bars
    exit_code = os.system('pip install -q -q timm albumentations facenet-pytorch > /dev/null 2>&1')
    
    if exit_code != 0:
        print("⚠️ Warning: Library installation might have failed (or just no output).")

    # 2. Run Training
    print("🚀 Starting Training (Wrapper Script)...")
    # Force PYTHONPATH to current directory
    # Using 'kaggle_convnext.yaml' (Quality Augmented Version)
    exit_code = os.system('PYTHONPATH=. python src/trainer/train.py --config configs/kaggle_convnext.yaml')
    
    if exit_code != 0:
        print("❌ Training Failed!")
        return

    # 3. Find and Display Checkpoint
    print("\n🔎 Searching for checkpoints...")
    ckpt_files = glob.glob('checkpoints/*.pt')
    
    if ckpt_files:
        try:
            from IPython.display import FileLink, display
            for f in ckpt_files:
                print(f"✅ Found: {f}")
                display(FileLink(f))
        except ImportError:
            print("⚠️ IPython not found. (Not running in Notebook?)")
            print(f"Checkpoints are at: {ckpt_files}")
    else:
        print("❌ No checkpoints found.")

if __name__ == '__main__':
    run_kaggle()