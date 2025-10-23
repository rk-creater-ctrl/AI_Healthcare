# ==============================
# File: test_and_report.py
# Purpose: Predict disease automatically based on modality and save PDF report
# ==============================

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tkinter import Tk, simpledialog
from tkinter.filedialog import askopenfilename
import cv2
import os
from datetime import datetime
from fpdf import FPDF
import json

# ==============================
# 1️⃣ Load models
# ==============================
print("🧠 Loading models...")

xray_ct_model = tf.keras.models.load_model("models/xray_ct_disease_model.h5")
mri_model = tf.keras.models.load_model("models/mri_disease_model.h5")
modality_model = tf.keras.models.load_model("models/modality_model.h5")

with open("models/xray_ct_class_names.json", "r") as f:
    xray_ct_classes = {v: k for k, v in json.load(f).items()}

with open("models/mri_class_names.json", "r") as f:
    mri_classes = {v: k for k, v in json.load(f).items()}

with open("models/modality_class_names.json", "r") as f:
    modality_classes = {v: k for k, v in json.load(f).items()}

print("✅ Models loaded successfully!")

# ==============================
# 2️⃣ Get patient info
# ==============================
Tk().withdraw()  # hide main window
patient_name = simpledialog.askstring("Patient Info", "Enter patient name:")
patient_age = simpledialog.askstring("Patient Info", "Enter patient age:")

if not patient_name or not patient_age:
    print("❌ Patient info incomplete. Exiting...")
    exit()

# ==============================
# 3️⃣ Select image
# ==============================
IMG_PATH = askopenfilename(
    title="Select Medical Image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
)
if not IMG_PATH:
    print("❌ No image selected. Exiting...")
    exit()

print(f"📷 Selected image: {IMG_PATH}")

# ==============================
# 4️⃣ Preprocess image
# ==============================
img = image.load_img(IMG_PATH, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array_exp = np.expand_dims(img_array, axis=0) / 255.0

# ==============================
# 5️⃣ Detect modality automatically
# ==============================
mod_pred = modality_model.predict(img_array_exp)
mod_idx = np.argmax(mod_pred[0])
modality = modality_classes[mod_idx]
print(f"🧪 Detected modality: {modality}")

# ==============================
# 6️⃣ Predict disease
# ==============================
if modality == "xray_ct":
    model = xray_ct_model
    classes = xray_ct_classes
else:
    model = mri_model
    classes = mri_classes

pred = model.predict(img_array_exp)
idx = np.argmax(pred[0])
confidence = pred[0][idx] * 100
label = classes[idx]

print(f"\n🩻 Prediction: {label} ({confidence:.2f}% confidence)\n")

# ==============================
# 7️⃣ Display image with label
# ==============================
img_cv = cv2.imread(IMG_PATH)
img_cv = cv2.resize(img_cv, (500, 500))
cv2.putText(
    img_cv,
    f"{label} ({confidence:.2f}%)",
    (10, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255) if confidence > 50 else (0, 255, 0),
    2
)
cv2.imshow("Prediction", img_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ==============================
# 8️⃣ Save PDF report
# ==============================
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
report_name = f"{patient_name}_{timestamp}.pdf"
report_path = os.path.join(reports_dir, report_name)

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", "B", 16)
pdf.cell(0, 10, "Medical Image Diagnosis Report", ln=True, align="C")
pdf.ln(10)

pdf.set_font("Arial", "", 12)
pdf.cell(0, 10, f"Patient Name: {patient_name}", ln=True)
pdf.cell(0, 10, f"Patient Age: {patient_age}", ln=True)
pdf.cell(0, 10, f"Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
pdf.cell(0, 10, f"Detected Modality: {modality}", ln=True)
pdf.cell(0, 10, f"Disease Prediction: {label} ({confidence:.2f}% confidence)", ln=True)
pdf.ln(10)
pdf.cell(0, 10, f"Image Path: {IMG_PATH}", ln=True)

pdf.output(report_path)
print(f"✅ Report saved at {report_path}")
