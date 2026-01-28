import opendatasets as od
import os
import shutil

# Make sure to handle credentials. 
# opendatasets looks for kaggle.json in the same folder or asks for input.
# Since we have ~/.kaggle/kaggle.json, we can read it and trick opendatasets or copy it.

def download_with_od():
    dataset_url = "https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces"
    data_dir = "data/raw"
    
    # Check if kaggle.json exists in home
    kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(kaggle_json_path):
        # opendatasets reads from current dir 'kaggle.json' if input not provided
        # or we can rely on it prompting... but we are in non-interactive mode.
        # It's better to manually copy kaggle.json to current dir for a moment.
        if not os.path.exists("kaggle.json"):
            shutil.copy(kaggle_json_path, "kaggle.json")
    
    try:
        od.download(dataset_url, data_dir)
        print("Download successful via opendatasets!")
    except Exception as e:
        print(f"OD Download failed: {e}")
    finally:
        # Clean up
        if os.path.exists("kaggle.json"):
            os.remove("kaggle.json")

if __name__ == "__main__":
    download_with_od()
