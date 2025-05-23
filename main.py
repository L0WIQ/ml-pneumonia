# Dépendances
import os
from PIL import Image

# Generate list of folder to
base_path = "./pneumonia"
folder_list = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

# Format folders
for folder in folder_list:
    # Parameters
    size = (224, 224)
    angle = 2

    # Normal images
    image_normal = os.listdir(f"{base_path}/{folder}/normal")

    # Format images
    for image in image_normal:
        im = Image.open(image)
        im.resize(size=size)
        for rotation in range(180):
            im.save(fp=(f"./new-date/{folder}/{image}"))
            im.rotate()

    # Pneumonia images
    image_pneumonia = os.listdir(f"{base_path}/{folder}/pneumonia")