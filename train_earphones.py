import os
from roboflow import Roboflow
from ultralytics import YOLO

# --- IMPORTANT SECURITY NOTE ---
# Your API key is a secret! Do not share it.
# I've put a placeholder here.
# You must replace "YOUR_API_KEY_HERE" with your actual key.
YOUR_API_KEY = "zb0lUz0thekklkhM7HDI"

print("Step 1: Downloading your 'earphone' dataset from Roboflow...")

# Authenticate with Roboflow
rf = Roboflow(api_key=YOUR_API_KEY)
project = rf.workspace("projects-opfhi").project("image-detector-huuut")
version = project.version(1)

# Download the dataset in YOLOv8 format
dataset = version.download("yolov8")

# Get the path to the data.yaml file
data_yaml_path = os.path.join(dataset.location, "data.yaml")

print(f"Dataset downloaded successfully to: {dataset.location}")
print(f"Using data.yaml from: {data_yaml_path}")

print("\nStep 2: Starting the model training...")

# Load the "Generalist" model (yolov8n.pt) to fine-tune
model = YOLO('yolov8n.pt')

# Start the training!
# We are 'fine-tuning' the model, so it learns 'earphone'
# in addition to the 80 objects it already knows.
results = model.train(
    data=data_yaml_path,  # Path to our new data.yaml
    epochs=50,            # 50 "rounds" of training. More is slower but often better.
    imgsz=640             # Image size
)

print("\nStep 3: Training complete!")
print("Your new custom model is saved in the 'runs/detect/train/' folder.")
print("The best model is called 'best.pt'.")