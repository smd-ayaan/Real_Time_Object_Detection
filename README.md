# Real-Time Object Detection (YOLOv8 & OpenCV)

Just a fun project I put together to learn more about computer vision. This is a Python script that uses your webcam to detect objects in real-time. 

I used the YOLOv8 Nano model because it's super lightweight—meaning it runs smoothly and gives high frame rates even if you don't have a massive GPU.

There are two main parts to this repo:
* `general_detector.py`: This is the main script. It uses the pre-trained YOLO model to detect standard, everyday objects live on camera.
* `train_earphones.py`: I wanted to figure out how to teach the model to recognize new things, so I wrote this script to fine-tune the model on a custom dataset to specifically detect earphones.

### How to use it
First, you'll need the right libraries:
`pip install opencv-python ultralytics`

To run the live webcam detector:
`python general_detector.py`

To run the custom training script:
`python train_earphones.py`
(Note: If you run the training script, make sure your custom dataset and labels are properly set up and linked in your data.yaml file).
