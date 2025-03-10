from ultralytics import YOLO
import cv2
from torchvision import models, transforms
from PIL import Image
import torch
import numpy as np
import pandas as pd
import pytesseract
import joblib  # For loading the pre-trained model

# Load CSV files
features_path = r"C:\Users\Lenova\Desktop\old dumper NN\features.csv"
predictions_path = r"C:\Users\Lenova\Desktop\old dumper NN\predictions.csv"
features_df = pd.read_csv(features_path)
predictions_df = pd.read_csv(predictions_path)

# Load YOLO model for truck detection
truck_model = YOLO("yolov8n.pt")

# Load ResNet50 for truck classification
resnet_model = models.resnet50(pretrained=True)
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, 2)  # 2 classes (Old/New)

# Load the pre-trained model for number plate classification
model_path = r"C:\Users\Lenova\Desktop\old dumper NN\models\model.pkl"
number_plate_model = joblib.load(model_path)

# Load the truck image
image_path = r"C:\Users\Lenova\Desktop\old dumper NN\dataset\testimg.jpeg"  # Replace with actual test image path
image = Image.open(image_path)

# Preprocess image for ResNet50 classification
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
img_tensor = transform(image).unsqueeze(0)  # Convert to tensor

# Classify truck as Old/New
resnet_model.eval()
with torch.no_grad():
    output = resnet_model(img_tensor)
predicted_class = torch.argmax(output).item()
truck_type = "Old" if predicted_class == 0 else "New"

# Convert image to OpenCV format
image_cv = cv2.imread(image_path)

# Perform truck detection
results = truck_model(image_path)

# Load YOLO model for License Plate Detection
license_plate_model = YOLO(r'C:\Users\Lenova\Desktop\old dumper NN\models\license_plate_detector.pt')  # Replace with actual path

# Function to extract features for number plate classification
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config="--psm 6")
    text_clarity = len(text.strip())
    edges = cv2.Canny(gray, 50, 150)
    edge_sharpness = np.sum(edges) / (gray.shape[0] * gray.shape[1])
    rust_level = 0  # Rust detection is not applicable for grayscale images
    return [rust_level, text_clarity, edge_sharpness]

number_plate_type = "Unknown"  # Default value

for result in results:
    for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
        TRUCK_CLASS_ID = 7
        if int(cls) == TRUCK_CLASS_ID:  # Check if detected object is a truck
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(image_cv, f"Truck ({truck_type})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Crop truck region and detect license plate
            truck_crop = image_cv[y1:y2, x1:x2]
            lp_results = license_plate_model(truck_crop)
            
            for lp_result in lp_results:
                for lp_box in lp_result.boxes.xyxy:
                    lp_x1, lp_y1, lp_x2, lp_y2 = map(int, lp_box[:4])
                    lp_x1 += x1
                    lp_y1 += y1
                    lp_x2 += x1
                    lp_y2 += y1
                    cv2.rectangle(image_cv, (lp_x1, lp_y1), (lp_x2, lp_y2), (255, 0, 0), 2)
                    cv2.putText(image_cv, "License Plate", (lp_x1, lp_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
                    # Extract license plate region
                    lp_crop = image_cv[lp_y1:lp_y2, lp_x1:lp_x2]
                    
                    # Extract features for number plate classification
                    features = extract_features(lp_crop)
                    features = np.array(features).reshape(1, -1)
                    
                    # Predict using the pre-trained model
                    number_plate_prediction = number_plate_model.predict(features)[0]
                    number_plate_type = "New" if number_plate_prediction == 1 else "Old"
                    
                    # Display the result on the image
                    cv2.putText(image_cv, f"Plate: {number_plate_type}", (lp_x1, lp_y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# Display the final image with all annotations
cv2.imshow("Truck and License Plate Detection", image_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Final output
print(f"Truck Type: {truck_type}")
print(f"Number Plate Type: {number_plate_type}")

# Fraud detection logic
if truck_type == "Old" and number_plate_type == "New":
    print("\033[91mOLD DUMPER WITH NEW NUMBER 🚨\033[0m")  # Red color warning
