from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create ImageDataGenerator with augmentation
datagen = ImageDataGenerator(
    rescale=1./255,          # normalize pixel values
    rotation_range=15,       # rotate images slightly
    zoom_range=0.1,          # zoom randomly
    horizontal_flip=True,    # flip horizontally
    validation_split=0.2     # 20% of data used for validation
)

# Training data loader
train_generator = datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# Validation data loader
val_generator = datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

print("✅ Data loaded successfully!")
print("Classes:", train_generator.class_indices)
