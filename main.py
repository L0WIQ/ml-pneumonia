# Dépendances
import os
from PIL import Image

# Generate list of folder to
base_path = "./pneumonia"
folder_list = [
    f for f in os.listdir(base_path)
    if os.path.isdir(os.path.join(base_path, f)) and f != ".git"
]

# Format folders
for folder in folder_list:
    # Parameters
    size = (224, 224)
    
    # Normal images
    image_normal = os.listdir(f"{base_path}/{folder}/normal")

    # Format images
    for image in image_normal:
        im = Image.open(f"{base_path}/{folder}/normal/{image}")
        dest_dir = f"./new-data/{folder}/normal"
        os.makedirs(dest_dir, exist_ok=True)
        im_resized = im.resize(size=size)
        angle = 0
        for rotation in range(4):
            im_rotated = im_resized.rotate(angle)
            dest_path = os.path.join(dest_dir, f"{angle}_{image}")
            im_rotated.save(dest_path)
            angle += 90

    # Pneumonia images
    image_pneumonia = os.listdir(f"{base_path}/{folder}/pneumonia")

    for image in image_pneumonia:
        im = Image.open(f"{base_path}/{folder}/pneumonia/{image}")
        dest_dir = f"./new-data/{folder}/pneumonia"
        os.makedirs(dest_dir, exist_ok=True)
        im_resized = im.resize(size=size)
        angle = 0
        for rotation in range(4):
            im_rotated = im_resized.rotate(angle)
            dest_path = os.path.join(dest_dir, f"{angle}_{image}")
            im_rotated.save(dest_path)
            angle += 90