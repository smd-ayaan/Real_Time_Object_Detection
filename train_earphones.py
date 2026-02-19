import os
from roboflow import Roboflow
from ultralytics import YOLO

YOUR_API_KEY = "YOUR_API_KEY"

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
results = model.train(
    data=data_yaml_path,  
    epochs=50,            
    imgsz=640             
)

print("\nStep 3: Training complete!")
print("Your new custom model is saved in the 'runs/detect/train/' folder.")
print("The best model is called 'best.pt'.")
