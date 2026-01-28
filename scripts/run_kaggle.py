import os
import glob

def run_kaggle():
    # 1. Run Training
    print("🚀 Starting Training (Wrapper Script)...")
    exit_code = os.system('python src/trainer/train.py --config configs/kaggle_convnext.yaml')
    
    if exit_code != 0:
        print("❌ Training Failed!")
        return

    # 2. Find and Display Checkpoint
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
