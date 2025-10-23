import os

train_path = "dataset/train"
val_path = "dataset/val"


# Count images
normal_train = len(os.listdir(os.path.join(train_path, "NORMAL")))
pneumonia_train = len(os.listdir(os.path.join(train_path, "PNEUMONIA")))

normal_val = len(os.listdir(os.path.join(val_path, "NORMAL")))
pneumonia_val = len(os.listdir(os.path.join(val_path, "PNEUMONIA")))

print(f"Training set -> NORMAL: {normal_train}, PNEUMONIA: {pneumonia_train}")
print(f"Validation set -> NORMAL: {normal_val}, PNEUMONIA: {pneumonia_val}")
