import os
from PIL import Image
from tqdm import tqdm

# CONFIG
IMAGES_DIR = "../data/processed/images"  # Adjust path relative to where you run the script

def verify_images():
    print(f"Checking images in {IMAGES_DIR}...")
    bad_files = []
    
    if not os.path.exists(IMAGES_DIR):
        print(f"Error: Directory {IMAGES_DIR} not found.")
        return

    files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for filename in tqdm(files):
        file_path = os.path.join(IMAGES_DIR, filename)
        try:
            with Image.open(file_path) as img:
                img.verify() # This checks for corruption
        except (IOError, SyntaxError) as e:
            print(f"Bad file found: {filename}")
            bad_files.append(file_path)

    if bad_files:
        print(f"\nFound {len(bad_files)} corrupted images.")
        # Optional: Uncomment to auto-delete
        # for f in bad_files:
        #     os.remove(f)
        # print("Deleted corrupted files.")
    else:
        print("\nAll images are valid!")

if __name__ == "__main__":
    verify_images()