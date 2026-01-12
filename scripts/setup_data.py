import os
import zipfile
# from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image
from tqdm import tqdm

# Config
DATASET_NAME = "paramaggarwal/fashion-product-images-dataset"
DOWNLOAD_PATH = "../data_temp"  # Temporary download location
RAW_IMAGES_PATH = "data/raw/images"
FINAL_IMAGES_PATH = "data/processed/images"
TARGET_SIZE = (512, 512) # Resize big images to this (High Quality, Low Size)

# def download_and_extract():
#     print("Authenticate with Kaggle...")
#     api = KaggleApi()
#     api.authenticate()

#     if not os.path.exists(DOWNLOAD_PATH):
#         os.makedirs(DOWNLOAD_PATH)

#     print(f"Downloading {DATASET_NAME} (This may take a while)...")
#     # This downloads the zip file
#     api.dataset_download_files(DATASET_NAME, path=DOWNLOAD_PATH, unzip=True)
#     print("Download and Extraction Complete.")

def process_images():
    print("Starting Image Optimization (Resizing)...")
    
    # The dataset usually extracts into a subfolder structure, 
    # typically: ./data_temp/fashion-dataset/images
    source_dir = os.path.join(RAW_IMAGES_PATH)
    
    if not os.path.exists(FINAL_IMAGES_PATH):
        os.makedirs(FINAL_IMAGES_PATH)

    files = [f for f in os.listdir(RAW_IMAGES_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for filename in tqdm(files):
        try:
            # 1. Open Image
            img_path = os.path.join(source_dir, filename)
            with Image.open(img_path) as img:
                # 2. Convert to RGB (Fixes issues with transparent PNGs if any)
                img = img.convert('RGB')
                
                # 3. Resize (Maintain aspect ratio or pad? For search, strict resize is usually ok)
                # But to be safe let's use thumbnail to keep aspect ratio
                img.thumbnail(TARGET_SIZE, Image.Resampling.LANCZOS)
                
                # 4. Save to final folder
                save_path = os.path.join(FINAL_IMAGES_PATH, filename)
                img.save(save_path, "JPEG", quality=85)
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("Image Optimization Complete.")

def cleanup():
    print("Cleaning up large temporary files...")
    # Add logic here to delete ./data_temp if you want to automate cleanup
    # shutil.rmtree(RAW_IMAGES_PATH)
    print(f"Done! Your optimized dataset is in {FINAL_IMAGES_PATH}")

if __name__ == "__main__":
    # download_and_extract()
    process_images()
    cleanup()