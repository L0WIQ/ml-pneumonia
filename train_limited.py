import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import os

# --- PARAMÈTRES ---
IMG_SIZE = (112, 112)
BATCH_SIZE = 8
DATASET_PATH = "pneumonia"

# --- 1. CHARGEMENT DES DONNÉES AVEC LIMITE ---
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=123,
    validation_split=0.9,  # on garde que 10%
    subset="validation"
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=123,
    validation_split=0.9,  # on garde que 10%
    subset="validation"
)

# --- 2. OPTIMISATION ---
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 3. VGG16 PRÉ-ENTRAÎNÉ ---
base_model = VGG16(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base_model.trainable = False

# --- 4. MODÈLE COMPLET ---
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

# --- 5. COMPILATION ---
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# --- 6. ENTRAÎNEMENT RÉDUIT ---
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,  # max 2 pour test rapide
    callbacks=[EarlyStopping(patience=1, restore_best_weights=True)]
)

model.save("pneumonie_model_limited.keras")