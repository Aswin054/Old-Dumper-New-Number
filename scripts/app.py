import os
import cv2
import torch
import joblib
import shutil
import asyncio
import streamlit as st
import nest_asyncio
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
import pytesseract

# ✅ Disable Streamlit File Watcher to Fix Async Issues
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# ✅ Fix asyncio error in Python 3.12+
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

nest_asyncio.apply()

# ✅ Define Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

features_path = os.path.join(DATA_DIR, "features.csv")
predictions_path = os.path.join(DATA_DIR, "predictions.csv")
model_path = os.path.join(MODEL_DIR, "model.pkl")
truck_model_path = os.path.join(MODEL_DIR, "yolov8n.pt")
license_plate_model_path = os.path.join(MODEL_DIR, "license_plate_detector.pt")

# ✅ Load CSV Safely
if os.path.exists(features_path) and os.path.exists(predictions_path):
    features_df = pd.read_csv(features_path)
    predictions_df = pd.read_csv(predictions_path)
else:
    st.warning("⚠️ Missing `features.csv` or `predictions.csv` in `data` folder!")

# ✅ Load YOLO Models Safely
truck_model, license_plate_model = None, None
try:
    truck_model = YOLO(truck_model_path)
    license_plate_model = YOLO(license_plate_model_path)
except Exception as e:
    st.error(f"⚠️ Error loading YOLO models: {str(e)}")

# ✅ Load ResNet50 Model Safely
resnet_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, 2)
resnet_model.eval()

# ✅ Load Number Plate Classifier Safely
number_plate_model = None
if os.path.exists(model_path):
    try:
        number_plate_model = joblib.load(model_path)
        st.success("✅ Number plate model loaded successfully!")
    except Exception as e:
        st.error(f"⚠️ Error loading `model.pkl`: {str(e)}")
else:
    st.error("⚠️ `model.pkl` not found in `models` folder!")

# ✅ Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    text = pytesseract.image_to_string(gray, config="--psm 8").strip()
    text_clarity = len(text)
    edges = cv2.Canny(gray, 50, 150)
    edge_sharpness = np.sum(edges) / (gray.shape[0] * gray.shape[1])
    return [0, text_clarity, edge_sharpness]

# ✅ Streamlit UI
st.title("🚛 OLD DUMPER WITH NEW NUMBER DETECTOR")
st.write("Upload a truck image to detect fraud.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image_cv = np.array(image)
    st.image(image, caption="Uploaded Image")

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
            for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                if int(cls) == 7:  # Truck Class ID
                    x1, y1, x2, y2 = map(int, box[:4])
                    truck_crop = image_cv[y1:y2, x1:x2]
                    if license_plate_model:
                        lp_results = license_plate_model(truck_crop)
                        for lp_result in lp_results:
                            for lp_box in lp_result.boxes.xyxy.cpu().numpy():
                                lp_x1, lp_y1, lp_x2, lp_y2 = map(int, lp_box[:4])
                                lp_crop = image_cv[lp_y1:lp_y2, lp_x1:lp_x2]
                                st.image(lp_crop, caption="Detected License Plate")
                                if number_plate_model:
                                    features = np.array(extract_features(lp_crop)).reshape(1, -1)
                                    number_plate_prediction = number_plate_model.predict(features)[0]
                                    number_plate_type = "New" if number_plate_prediction == 1 else "Old"
    
    # ✅ Display Results
    st.subheader("Results:")
    st.write(f"**Truck Type:** {truck_type}")
    st.write(f"**Number Plate Type:** {number_plate_type}")
    
    # ✅ Fraud Detection
    if truck_type == "Old" and number_plate_type == "New":
        st.error("🚨 OLD DUMPER WITH NEW NUMBER DETECTED 🚨")
    else:
        st.success("✅ No fraud detected")
