import os

# ✅ Disable Streamlit File Watcher to Fix Torch Async Issues
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import asyncio
import streamlit as st
import nest_asyncio

# ✅ Fix asyncio error in Python 3.12+
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

nest_asyncio.apply()

# ✅ Import Libraries
import cv2
import numpy as np
import torch
import joblib
import pandas as pd
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from ultralytics import YOLO
import pytesseract

# ✅ Define Paths (Fix Deployment Issue)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Go one level up

MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")  # Store CSV files in a `data/` folder for better organization

truck_model_path = os.path.join(MODEL_DIR, "yolov8n.pt")
license_plate_model_path = os.path.join(MODEL_DIR, "license_plate_detector.pt")
number_plate_model_path = os.path.join(MODEL_DIR, "model.pkl")

FEATURES_CSV = os.path.join(DATA_DIR, "features.csv")
PREDICTIONS_CSV = os.path.join(DATA_DIR, "predictions.csv")

# ✅ Load YOLO Models (Check if files exist)
if os.path.exists(truck_model_path):
    truck_model = YOLO(truck_model_path)
else:
    st.error("⚠️ `yolov8n.pt` model not found in the `models/` directory!")

if os.path.exists(license_plate_model_path):
    license_plate_model = YOLO(license_plate_model_path)
else:
    st.error("⚠️ `license_plate_detector.pt` model not found in the `models/` directory!")

# ✅ Load Number Plate Classifier (Handle Missing File Safely)
number_plate_model = None
if os.path.exists(number_plate_model_path):
    try:
        number_plate_model = joblib.load(number_plate_model_path)
    except Exception as e:
        st.error(f"⚠️ Error loading `model.pkl`: {str(e)}")
else:
    st.error("⚠️ `model.pkl` not found in the `models/` directory!")

# ✅ Load CSV files (Handle Missing Files)
try:
    features_df = pd.read_csv(FEATURES_CSV)
    predictions_df = pd.read_csv(PREDICTIONS_CSV)
    st.success("✅ CSV Files Loaded Successfully!")
except FileNotFoundError:
    st.error("⚠️ Missing CSV files: `features.csv` or `predictions.csv` not found!")

# ✅ Fix ResNet50 model loading (removes warning)
resnet_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, 2)
resnet_model.eval()

# ✅ Image Preprocessing for ResNet50
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_features(image):
    """Extract features from the license plate for classification."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config="--psm 8")
    text_clarity = len(text.strip())

    edges = cv2.Canny(gray, 50, 150)
    edge_sharpness = np.sum(edges) / (gray.shape[0] * gray.shape[1])

    rust_level = 0  # No rust detection in grayscale images
    return [rust_level, text_clarity, edge_sharpness]

# ✅ Streamlit UI
st.title("🚛 OLD DUMPER WITH NEW NUMBER DETECTOR")
st.write("Upload an image of a truck to detect its type and check for fraud.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # ✅ Read and display the image
    image = Image.open(uploaded_file)
    image_cv = np.array(image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ✅ Truck Classification
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = resnet_model(img_tensor)
    truck_type = "Old" if torch.argmax(output).item() == 0 else "New"

    # ✅ Truck Detection
    results = truck_model(image_cv) if 'truck_model' in locals() else []
    number_plate_type = "Unknown"

    for result in results:
        detected_classes = set([int(cls) for cls in result.boxes.cls])
        st.write(f"Detected Classes in Image: {detected_classes}")  # Debugging output

        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            TRUCK_CLASS_ID = 7
            if int(cls) == TRUCK_CLASS_ID:
                x1, y1, x2, y2 = map(int, box[:4])
                truck_crop = image_cv[y1:y2, x1:x2]

                # ✅ License Plate Detection
                lp_results = license_plate_model(truck_crop) if 'license_plate_model' in locals() else []
                if not lp_results:
                    st.warning("⚠️ No license plate detected! Please try another image.")
                    continue

                for lp_result in lp_results:
                    for lp_box in lp_result.boxes.xyxy:
                        lp_x1, lp_y1, lp_x2, lp_y2 = map(int, lp_box[:4])
                        lp_x1 += x1
                        lp_y1 += y1
                        lp_x2 += x1
                        lp_y2 += y1
                        lp_crop = image_cv[lp_y1:lp_y2, lp_x1:lp_x2]
                        st.image(lp_crop, caption="Detected License Plate", use_container_width=True)

                        # ✅ Extract Features and Classify Number Plate
                        if number_plate_model is not None:
                            features = np.array(extract_features(lp_crop)).reshape(1, -1)
                            if np.isnan(features).any():
                                st.error("⚠️ Error in extracted features: NaN values found.")
                            else:
                                number_plate_prediction = number_plate_model.predict(features)[0]
                                number_plate_type = "New" if number_plate_prediction == 1 else "Old"
                        else:
                            st.error("⚠️ Number plate classifier model is missing!")

    # ✅ Display Results
    st.subheader("Results:")
    st.write(f"**Truck Type:** {truck_type}")
    st.write(f"**Number Plate Type:** {number_plate_type}")

    # ✅ Fraud Detection
    if truck_type == "Old" and number_plate_type == "New":
        st.error("🚨 OLD DUMPER WITH NEW NUMBER DETECTED 🚨")
    else:
        st.success("✅ No fraud detected")
