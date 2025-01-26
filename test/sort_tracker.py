import cv2
import numpy as np
import torch
from ultralytics import YOLO
import sys

# Add SORT directory path
sys.path.append(r"E:\RoboCam\RoboCam\assets\github_repos\sort")
from sort import Sort  # Import SORT tracker

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load YOLOv8 detection model
yolo_model = YOLO("../models/yolov8n.pt").to(device)  # YOLOv8 detection model

# Initialize SORT Tracker
sort_tracker = Sort()

# Load video file
video_path = r"E:\RoboCam\RoboCam\assets\dataset\Anomaly-Videos-Part-1\Assault\Assault001_x264.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 Detection on GPU
    results = yolo_model(frame, device=device)
    detections = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
    confidences = results[0].boxes.conf.cpu().numpy()  # Get confidence scores
    classes = results[0].boxes.cls.cpu().numpy()  # Get class IDs

    # Filter only "person" detections (COCO class ID 0)
    person_detections = []
    for box, conf, cls in zip(detections, confidences, classes):
        if int(cls) == 0:  # Class 0 is 'person' in COCO
            x1, y1, x2, y2 = map(int, box)
            person_detections.append([x1, y1, x2, y2, conf])  # Format for SORT: [x1, y1, x2, y2, confidence]

    # Convert detections to NumPy array with correct shape
    if len(person_detections) > 0:
        person_detections = np.array(person_detections)
    else:
        person_detections = np.empty((0, 5))  # Ensure correct shape when empty

    # Update SORT tracker
    tracked_objects = sort_tracker.update(person_detections)

    # Draw bounding boxes with tracking IDs
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, obj)
        color = (track_id * 30 % 256, track_id * 60 % 256, track_id * 90 % 256)  # Assign unique color for each ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display the frame
    cv2.imshow("YOLOv8 + SORT Tracking", frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()