import os
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset():
    api = KaggleApi()
    api.authenticate()
    
    dataset = "xhlulu/140k-real-and-fake-faces"
    path = "data/raw"
    
    print(f"Downloading {dataset} to {path}...")
    try:
        # CLI wrapper sometimes has bugs, calling API directly
        api.dataset_download_files(dataset, path=path, unzip=True)
        print("Download complete!")
    except Exception as e:
        print(f"Download failed: {e}")

if __name__ == "__main__":
    download_dataset()
