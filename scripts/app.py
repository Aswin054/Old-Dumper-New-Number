import cv2
print("✅ OpenCV Version:", cv2.__version__)


import os
import numpy as np
import torch
import joblib
import streamlit as st
import pandas as pd
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO
import pytesseract
import asyncio
import nest_asyncio


# ✅ **Disable Streamlit File Watcher (Fix Async Issues)**
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# ✅ **Fix asyncio error in Python 3.12+**
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

nest_asyncio.apply()

import shutil


# Automatically find Tesseract installation path
tesseract_path = shutil.which("tesseract")

if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    print(f"Tesseract found at: {tesseract_path}")

else:
    # Check if 'packages.txt' exists and contains 'tesseract'
    package_file = "packages.txt"
    if os.path.exists(package_file):
        with open(package_file, "r") as f:
            installed_packages = f.read()

        if "tesseract" in installed_packages.lower():
            print("Tesseract is listed in packages.txt but not found in PATH.")
            print("Try adding it to PATH or reinstalling.")
        else:
            print("Tesseract is missing. Please install it.")

    else:
        print("packages.txt not found. Cannot verify Tesseract installation.")
        print("Please install Tesseract-OCR and add it to your system PATH.")

# ✅ **Define Paths**
# ✅ Get the script directory (where the current script is running)
SCRIPT_DIR = os.path.dirname(__file__)

# ✅ Set the models folder inside the scripts directory
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")

# ✅ Get the project root (one level up from scripts)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# ✅ Set the data folder correctly in the project root
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

truck_model_path = os.path.join(MODEL_DIR, "yolov8n.pt")
license_plate_model_path = os.path.join(MODEL_DIR, "license_plate_detector.pt")
number_plate_model_path = os.path.join(MODEL_DIR, "model.pkl")

FEATURES_CSV = os.path.join(DATA_DIR, "features.csv")
PREDICTIONS_CSV = os.path.join(DATA_DIR, "predictions.csv")

# ✅ **Load YOLO Models Safely**
truck_model = YOLO(truck_model_path) if os.path.exists(truck_model_path) else None
license_plate_model = YOLO(license_plate_model_path) if os.path.exists(license_plate_model_path) else None

if not truck_model:
    st.error(f"⚠️ Truck detection model not found: {truck_model_path}")
if not license_plate_model:
    st.error(f"⚠️ License plate detection model not found: {license_plate_model_path}")

# ✅ **Load Number Plate Classifier**
number_plate_model = None
if os.path.exists(number_plate_model_path):
    try:
        number_plate_model = joblib.load(number_plate_model_path)
    except Exception as e:
        st.error(f"⚠️ Error loading model.pkl: {str(e)}")

# ✅ **Load ResNet50**
resnet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, 2)
resnet_model.eval()

# ✅ **Image Preprocessing**
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_features(image):
    """Extracts features from license plate."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config="--psm 8")
        text_clarity = len(text.strip())
        edges = cv2.Canny(gray, 50, 150)
        edge_sharpness = np.sum(edges) / (gray.shape[0] * gray.shape[1])
        rust_level = 0  # No rust detection in grayscale images
        return [rust_level, text_clarity, edge_sharpness]
    except Exception as e:
        st.error(f"⚠️ Error extracting features: {e}")
        return [0, 0, 0]

# ✅ **Streamlit UI**
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
    number_plate_type = "Unknown"
    
    if truck_model:
        results = truck_model(image_cv)

        for result in results:
            detected_classes = set([int(cls) for cls in result.boxes.cls])
            st.write(f"Detected Classes in Image: {detected_classes}")

            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                TRUCK_CLASS_ID = 7
                if int(cls) == TRUCK_CLASS_ID:
                    x1, y1, x2, y2 = map(int, box[:4])
                    truck_crop = image_cv[y1:y2, x1:x2]

                    # ✅ License Plate Detection
                    if license_plate_model:
                        lp_results = license_plate_model(truck_crop)

                        if lp_results:
                            for lp_result in lp_results:
                                for lp_box in lp_result.boxes.xyxy:
                                    lp_x1, lp_y1, lp_x2, lp_y2 = map(int, lp_box[:4])
                                    lp_x1 += x1
                                    lp_y1 += y1
                                    lp_x2 += x1
                                    lp_y2 += y1
                                    lp_crop = image_cv[lp_y1:lp_y2, lp_x1:lp_x2]
                                    st.image(lp_crop, caption="Detected License Plate", use_container_width=True)

                                    # ✅ Extract Features & Classify Number Plate
                                    if number_plate_model is not None:
                                        features = np.array(extract_features(lp_crop)).reshape(1, -1)
                                        feature_columns = ["Rust_Level", "Text_Clarity", "Edge_Sharpness"]  # Match training names

                                        features_df = pd.DataFrame(features, columns=feature_columns)

                                        if np.isnan(features).any():
                                            st.error("Error in extracted features: NaN values found.")
                                        else:
                                            number_plate_prediction = number_plate_model.predict(features_df)[0]
                                            number_plate_type = "New" if number_plate_prediction == 1 else "Old"
                                    else:
                                        st.error("⚠️ Number plate classifier model is missing!")
                        else:
                            st.warning("⚠️ No license plate detected! Please try another image.")
                    else:
                        st.error("⚠️ License plate detection model not loaded!")

    # ✅ Display Results
    st.subheader("Results:")
    st.write(f"**Truck Type:** {truck_type}")
    st.write(f"**Number Plate Type:** {number_plate_type}")

    # ✅ Fraud Detection
    if truck_type == "Old" and number_plate_type == "New":
        st.error("🚨 OLD DUMPER WITH NEW NUMBER DETECTED 🚨")
    else:
        st.success("✅ No fraud detected")
