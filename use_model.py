import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# --- PARAMÈTRES ---
MODEL_PATH = "pneumonie_model_limited.keras"
IMG_PATH = sys.argv[1] if len(sys.argv) > 1 else "./pneumonia/test/NORMAL/NORMAL2-IM-0357-0001.jpeg"
IMG_SIZE = (112, 112)

# --- CHARGER LE MODÈLE ---
print(f"Chargement du modèle depuis : {MODEL_PATH}")
model = load_model(MODEL_PATH)

# --- CHARGER ET PRÉTRAITER L'IMAGE ---
print(f"Chargement de l’image : {IMG_PATH}")
img = image.load_img(IMG_PATH, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # normalisation

# --- PRÉDICTION ---
prediction = model.predict(img_array)[0][0]

# --- RESULTATS ---
print(f"score : {prediction:.2f}")
