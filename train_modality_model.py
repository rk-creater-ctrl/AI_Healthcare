import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import os

# -------------------
# Paths
# -------------------
DATASET_DIR = "dataset/modality"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------
# Data generator
# -------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
        rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(224,224),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(224,224),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# -------------------
# Model
# -------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(len(train_gen.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# -------------------
# Train
# -------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5  # start with 5, increase later
)

# -------------------
# Save
# -------------------
model.save(os.path.join(MODEL_DIR, "modality_model.h5"))
print("✅ Modality detection model trained and saved!")
