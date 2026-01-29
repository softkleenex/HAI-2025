import os
import glob
import yaml

def run_kaggle_celeb():
    # 1. Install Libraries (Silently)
    print("📦 Installing libraries (Silently)...")
    os.system('pip install -q -q timm albumentations facenet-pytorch kagglehub > /dev/null 2>&1')
    
    # 2. Download Celeb-DF via KaggleHub
    print("⬇️ Downloading Celeb-DF v2...")
    import kagglehub
    celeb_path = kagglehub.dataset_download("reubensuju/celeb-df-v2")
    print(f"✅ Celeb-DF downloaded to: {celeb_path}")
    
    # 3. Update Config Dynamically
    config_path = 'configs/kaggle_celeb_df.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update train_dir with actual downloaded path
    # Assuming FF++ is already attached via UI at /kaggle/input/...
    # If not, we might need to download FF++ too or assume it's there.
    # Let's assume FF++ is attached via UI (Standard) and Celeb is via Hub.
    
    # Clean up the placeholder path for Celeb-DF and insert the real one
    new_train_dirs = []
    for d in config['data']['train_dir']:
        if 'celeb-df' in d.lower():
            new_train_dirs.append(celeb_path) # Replace with real path
        else:
            new_train_dirs.append(d) # Keep FF++ path
            
    config['data']['train_dir'] = new_train_dirs
    
    # Save temp config
    temp_config = 'configs/temp_kaggle_celeb.yaml'
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
        
    print(f"⚙️ Updated Config: {config['data']['train_dir']}")

    # 4. Run Training
    print("🚀 Starting Generalist Training (FF++ & Celeb-DF)...")
    exit_code = os.system(f'PYTHONPATH=. python src/trainer/train.py --config {temp_config}')
    
    if exit_code != 0:
        print("❌ Training Failed!")
        return

    # 5. Find and Display Checkpoint
    print("\n🔎 Searching for checkpoints...")
    ckpt_files = glob.glob('checkpoints/*.pt')
    
    if ckpt_files:
        try:
            from IPython.display import FileLink, display
            for f in ckpt_files:
                print(f"✅ Found: {f}")
                display(FileLink(f))
        except ImportError:
            print(f"Checkpoints are at: {ckpt_files}")
    else:
        print("❌ No checkpoints found.")

if __name__ == '__main__':
    run_kaggle_celeb()
