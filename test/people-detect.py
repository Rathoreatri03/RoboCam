import cv2
import numpy as np
from ultralytics import YOLO
import torch

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load YOLOv8 models on GPU
yolo_detection = YOLO("../models/yolov8n.pt").to(device)  # Detection model
yolo_segmentation = YOLO("../models/yolov8n-seg.pt").to(device)  # Segmentation model

# IOU calculation function
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

# Load video file
video_path = r"E:\RoboCam\RoboCam\assets\dataset\Anomaly-Videos-Part-1\Assault\Assault001_x264.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 Detection on GPU
    results_det = yolo_detection(frame, device=device)
    detections = results_det[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
    det_classes = results_det[0].boxes.cls.cpu().numpy()  # Class IDs

    # YOLOv8-Segmentation on GPU
    results_seg = yolo_segmentation(frame, device=device)
    seg_boxes = results_seg[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
    seg_classes = results_seg[0].boxes.cls.cpu().numpy()  # Class IDs

    # Draw detection boxes (green)
    for box, cls in zip(detections, det_classes):
        if int(cls) == 0:  # Class 0 is 'person' in COCO
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw segmentation boxes (blue)
    for box, cls in zip(seg_boxes, seg_classes):
        if int(cls) == 0:  # Class 0 is 'person' in COCO
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Highlight overlapping boxes (yellow)
    for det_box in detections:
        for seg_box in seg_boxes:
            iou = compute_iou(det_box, seg_box)
            if iou > 0.5:  # IOU threshold
                x1, y1, x2, y2 = map(int, det_box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # Display the frame
    cv2.imshow("YOLOv8 Detection & Segmentation (GPU)", frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
