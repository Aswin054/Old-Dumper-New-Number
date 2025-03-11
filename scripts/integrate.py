import os
import cv2
import numpy as np
import torch
import pandas as pd
import joblib
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO
import pytesseract

# ✅ Automatically detect Tesseract path (same as app.py)
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if not os.path.exists(TESSERACT_PATH):
    raise FileNotFoundError(f"⚠️ Tesseract not found at {TESSERACT_PATH}. Check if it's installed correctly.")

# ✅ Assign to pytesseract
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# ✅ Define paths for models & data (same as app.py)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Project root
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = os.path.join(BASE_DIR, "data")  # CSV files stored here

truck_model_path = os.path.join(MODEL_DIR, "yolov8n.pt")
license_plate_model_path = os.path.join(MODEL_DIR, "license_plate_detector.pt")
number_plate_model_path = os.path.join(MODEL_DIR, "model.pkl")

FEATURES_CSV = os.path.join(DATA_DIR, "features.csv")
PREDICTIONS_CSV = os.path.join(DATA_DIR, "predictions.csv")

# ✅ Load CSV files safely
try:
    features_df = pd.read_csv(FEATURES_CSV)
    predictions_df = pd.read_csv(PREDICTIONS_CSV)
except FileNotFoundError:
    raise FileNotFoundError("⚠️ Missing CSV files: `features.csv` or `predictions.csv` not found!")

# ✅ Load YOLO models (same as app.py)
truck_model = YOLO(truck_model_path) if os.path.exists(truck_model_path) else None
license_plate_model = YOLO(license_plate_model_path) if os.path.exists(license_plate_model_path) else None

if not truck_model:
    raise FileNotFoundError(f"⚠️ Truck detection model not found: {truck_model_path}")
if not license_plate_model:
    raise FileNotFoundError(f"⚠️ License plate detection model not found: {license_plate_model_path}")

# ✅ Load Number Plate Classifier
number_plate_model = None
if os.path.exists(number_plate_model_path):
    try:
        number_plate_model = joblib.load(number_plate_model_path)
    except Exception as e:
        raise RuntimeError(f"⚠️ Error loading model.pkl: {str(e)}")

# ✅ Load ResNet50 for truck classification
resnet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, 2)
resnet_model.eval()

# ✅ Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ✅ Feature extraction (same as app.py)
def extract_features(image):
    """Extract features from license plate."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config="--psm 8")
        text_clarity = len(text.strip())
        edges = cv2.Canny(gray, 50, 150)
        edge_sharpness = np.sum(edges) / (gray.shape[0] * gray.shape[1])
        rust_level = 0  # No rust detection in grayscale images
        return [rust_level, text_clarity, edge_sharpness]
    except Exception as e:
        print(f"⚠️ Error extracting features: {e}")
        return [0, 0, 0]

# ✅ Load Test Image (Replace with actual path)
image_path = os.path.join(BASE_DIR, "dataset", "testimg.jpeg")
if not os.path.exists(image_path):
    raise FileNotFoundError(f"⚠️ Test image not found: {image_path}")

image = Image.open(image_path)

# ✅ Truck Classification
img_tensor = transform(image).unsqueeze(0)
with torch.no_grad():
    output = resnet_model(img_tensor)
truck_type = "Old" if torch.argmax(output).item() == 0 else "New"

# ✅ Truck Detection
image_cv = cv2.imread(image_path)
results = truck_model(image_cv) if truck_model else []

number_plate_type = "Unknown"

for result in results:
    for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
        TRUCK_CLASS_ID = 7
        if int(cls) == TRUCK_CLASS_ID:  
            x1, y1, x2, y2 = map(int, box[:4])
            truck_crop = image_cv[y1:y2, x1:x2]

            # ✅ License Plate Detection
            lp_results = license_plate_model(truck_crop) if license_plate_model else []

            for lp_result in lp_results:
                for lp_box in lp_result.boxes.xyxy:
                    lp_x1, lp_y1, lp_x2, lp_y2 = map(int, lp_box[:4])
                    lp_x1 += x1
                    lp_y1 += y1
                    lp_x2 += x1
                    lp_y2 += y1
                    lp_crop = image_cv[lp_y1:lp_y2, lp_x1:lp_x2]

                    # ✅ Extract Features and Classify Number Plate
                    if number_plate_model is not None:
                        features = np.array(extract_features(lp_crop)).reshape(1, -1)
                        feature_columns = ["Rust_Level", "Text_Clarity", "Edge_Sharpness"]  # Match training names
                                        
                        features_df = pd.DataFrame(features, columns=feature_columns)

                        

                        if np.isnan(features).any():
                            print("⚠️ Error: NaN values found in extracted features.")
                        else:
                            number_plate_prediction = number_plate_model.predict(features_df)[0]
                            number_plate_type = "New" if number_plate_prediction == 1 else "Old"
                    else:
                        print("⚠️ Number plate classifier model is missing!")

# ✅ Display Results
print(f"Truck Type: {truck_type}")
print(f"Number Plate Type: {number_plate_type}")

# ✅ Fraud Detection
if truck_type == "Old" and number_plate_type == "New":
    print("\033[91m🚨 OLD DUMPER WITH NEW NUMBER DETECTED! 🚨\033[0m")
else:
    print("✅ No fraud detected")
