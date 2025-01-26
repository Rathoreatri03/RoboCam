import cv2
from ultralytics import YOLO
import torch
# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# Load YOLOv8 models on GPU
yolo_detection = YOLO("../models/yolov8n.pt").to(device)  # Detection model
yolo_segmentation = YOLO("../models/yolov8n-seg.pt").to(device)  # Segmentation model

video_path = r"E:\RoboCam\RoboCam\assets\dataset\Anomaly-Videos-Part-1\Abuse\Abuse002_x264.mp4"
cap = cv2.VideoCapture(video_path)

