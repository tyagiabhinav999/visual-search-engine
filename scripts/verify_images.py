from PIL import Image
import os

def check_images(folder_path):
    valid_images = []
    for filename in os.listdir(folder_path):
        try:
            with Image.open(os.path.join(folder_path, filename)) as img:
                img.verify() # Checks if file is broken
            valid_images.append(filename)
        except:
            print(f"Bad file: {filename}")
    return valid_images

# Run this once on your 'images' folder to get a clean list