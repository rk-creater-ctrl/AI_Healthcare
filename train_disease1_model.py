# ================================
# File: train_disease_model.py
# Purpose: Train disease classification model for X-ray + CT
# ================================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import os
from PIL import ImageFile

# ================================
# 🧩 Fix for corrupted images
# ================================
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Prevent OSError: image file is truncated

# ================================
# 1️⃣ Paths
# ================================
train_path = "dataset/disease/mri/train"
val_path   = "dataset/disease/mri/val"
models_dir = "models"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# ================================
# 2️⃣ Data generators with augmentation
# ================================
datagen_train = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

datagen_val = ImageDataGenerator(rescale=1.0 / 255)

train_generator = datagen_train.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode="categorical"
)

val_generator = datagen_val.flow_from_directory(
    val_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode="categorical"
)

print(f"✅ Classes found: {train_generator.class_indices}")

# ================================
# 3️⃣ Create model (MobileNetV2 base)
# ================================
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
predictions = Dense(train_generator.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile
model.compile(
    optimizer=Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ================================
# 4️⃣ Train model
# ================================
epochs = 10  # you can increase to 20–30 later for better accuracy

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)

# ================================
# 5️⃣ Save model
# ================================
model_path = os.path.join(models_dir, "mri_disease_model.h5")
model.save(model_path)
print(f"✅ Model trained and saved successfully at {model_path}!")
