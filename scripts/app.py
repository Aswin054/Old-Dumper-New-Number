import os
import asyncio
import shutil
import streamlit as st
import nest_asyncio
import numpy as np
import torch
import joblib
import pandas as pd
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
import pytesseract

# ✅ Fix OpenCV Import Issue
try:
    import cv2
except ImportError:
    st.error("⚠️ OpenCV import failed! Ensure `opencv-python-headless` is installed.")

# ✅ Fix YOLO Import Issue
try:
    from ultralytics import YOLO
except ImportError:
    st.error("⚠️ Ultralytics YOLO import failed! Try reinstalling with `pip install ultralytics`.")

# ✅ Disable Streamlit File Watcher (Fixes Torch Issues)
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# ✅ Fix asyncio issues in Python 3.12+
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

nest_asyncio.apply()

# ✅ Auto-detect Tesseract path
tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    st.error("⚠️ Tesseract-OCR not found! Ensure it's installed and added to system PATH.")

# ✅ Define Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

truck_model_path = os.path.join(MODEL_DIR, "yolov8n.pt")
license_plate_model_path = os.path.join(MODEL_DIR, "license_plate_detector.pt")
number_plate_model_path = os.path.join(MODEL_DIR, "model.pkl")

FEATURES_CSV = os.path.join(DATA_DIR, "features.csv")
PREDICTIONS_CSV = os.path.join(DATA_DIR, "predictions.csv")

# ✅ Load YOLO Models (Handle Errors)
truck_model, license_plate_model = None, None
try:
    if os.path.exists(truck_model_path):
        truck_model = YOLO(truck_model_path)
    if os.path.exists(license_plate_model_path):
        license_plate_model = YOLO(license_plate_model_path)
except Exception as e:
    st.error(f"⚠️ Error loading YOLO models: {str(e)}")

# ✅ Load Number Plate Classifier
number_plate_model = None
if os.path.exists(number_plate_model_path):
    try:
        number_plate_model = joblib.load(number_plate_model_path)
        st.success("✅ Number plate model loaded successfully!")
    except Exception as e:
        st.error(f"⚠️ Error loading `model.pkl`: {str(e)}")
else:
    st.error("⚠️ `model.pkl` not found!")

# ✅ Load CSV Files
try:
    features_df = pd.read_csv(FEATURES_CSV)
    predictions_df = pd.read_csv(PREDICTIONS_CSV)
    st.success("✅ CSV Files Loaded Successfully!")
except FileNotFoundError:
    st.warning("⚠️ Missing `features.csv` or `predictions.csv`!")

# ✅ Load ResNet50 Model
resnet_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, 2)
resnet_model.eval()

# ✅ Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_features(image):
    """Extract features from the license plate."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    try:
        text = pytesseract.image_to_string(gray, config="--psm 8").strip()
        text_clarity = len(text)
    except pytesseract.pytesseract.TesseractNotFoundError:
        st.error("⚠️ Tesseract-OCR not installed!")
        return [0, 0, 0]
    edges = cv2.Canny(gray, 50, 150)
    edge_sharpness = np.sum(edges) / (gray.shape[0] * gray.shape[1])
    rust_level = 0  # Placeholder for rust detection
    return [rust_level, text_clarity, edge_sharpness]

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
            detected_classes = set([int(cls) for cls in result.boxes.cls.cpu().numpy()])
            for box, cls in zip(result.boxes.xyxy.cpu().numpy(), detected_classes):
                if cls == 7:  # Truck Class ID
                    x1, y1, x2, y2 = map(int, box[:4])
                    truck_crop = image_cv[y1:y2, x1:x2]
                    if license_plate_model:
                        lp_results = license_plate_model(truck_crop)
                        for lp_result in lp_results:
                            for lp_box in lp_result.boxes.xyxy.cpu().numpy():
                                lp_x1, lp_y1, lp_x2, lp_y2 = map(int, lp_box[:4])
                                lp_x1 += x1
                                lp_y1 += y1
                                lp_x2 += x1
                                lp_y2 += y1
                                lp_crop = image_cv[lp_y1:lp_y2, lp_x1:lp_x2]
                                st.image(lp_crop, caption="Detected License Plate")
                                if number_plate_model:
                                    features = np.array(extract_features(lp_crop)).reshape(1, -1)
                                    feature_columns = ["Rust_Level", "Text_Clarity", "Edge_Sharpness"]
                                    features_df = pd.DataFrame(features, columns=feature_columns)
                                    if not np.isnan(features_df.values).any():
                                        number_plate_prediction = number_plate_model.predict(features_df)[0]
                                        number_plate_type = "New" if number_plate_prediction == 1 else "Old"
                                    else:
                                        st.error("⚠️ Extracted features contain NaN values!")

    # ✅ Display Results
    st.subheader("Results:")
    st.write(f"**Truck Type:** {truck_type}")
    st.write(f"**Number Plate Type:** {number_plate_type}")
    if truck_type == "Old" and number_plate_type == "New":
        st.error("🚨 OLD DUMPER WITH NEW NUMBER DETECTED 🚨")
    else:
        st.success("✅ No fraud detected")