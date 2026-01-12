import os
from PIL import Image
from tqdm import tqdm
from src.config import PROCESSED_IMAGES_DIR

def verify_images():
    print(f"Checking images in {PROCESSED_IMAGES_DIR}...")
    bad_files = []
    
    if not os.path.exists(PROCESSED_IMAGES_DIR):
        print(f"Error: Directory {PROCESSED_IMAGES_DIR} not found.")
        return

    files = [f for f in os.listdir(PROCESSED_IMAGES_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for filename in tqdm(files):
        file_path = os.path.join(PROCESSED_IMAGES_DIR, filename)
        try:
            with Image.open(file_path) as img:
                img.verify() 
        except (IOError, SyntaxError):
            print(f"Bad file found: {filename}")
            bad_files.append(file_path)

    if bad_files:
        print(f"\nFound {len(bad_files)} corrupted images.")
        # to auto-delete
        # for f in bad_files:
        #     os.remove(f)
        # print("Deleted corrupted files.")
    else:
        print("\nAll images are valid!")

if __name__ == "__main__":
    verify_images()