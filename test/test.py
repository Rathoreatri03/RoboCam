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
    det_scores = results_det[0].boxes.conf.cpu().numpy()  # Confidence scores

    # YOLOv8-Segmentation on GPU
    results_seg = yolo_segmentation(frame, device=device)
    seg_boxes = results_seg[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
    seg_classes = results_seg[0].boxes.cls.cpu().numpy()  # Class IDs
    seg_scores = results_seg[0].boxes.conf.cpu().numpy()  # Confidence scores

    # Combine the detection and segmentation results
    combined_boxes = []
    combined_scores = []
    combined_classes = []

    # Combine boxes from both models and keep track of confidence
    for box, cls, score in zip(detections, det_classes, det_scores):
        if int(cls) == 0:  # Class 0 is 'person' in COCO
            combined_boxes.append(box)
            combined_scores.append(score)
            combined_classes.append(cls)

    for box, cls, score in zip(seg_boxes, seg_classes, seg_scores):
        if int(cls) == 0:  # Class 0 is 'person' in COCO
            # Check if the same person is detected by both models, combine them
            overlap_found = False
            for i, combined_box in enumerate(combined_boxes):
                iou = compute_iou(box, combined_box)
                if iou > 0.5:  # IOU threshold
                    combined_scores[i] = max(combined_scores[i], score)  # Take the highest confidence
                    overlap_found = True
                    break
            if not overlap_found:
                combined_boxes.append(box)
                combined_scores.append(score)
                combined_classes.append(cls)

    # Draw the combined boxes (yellow) based on highest confidence
    for box, score in zip(combined_boxes, combined_scores):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 255)  # Yellow for combined detection
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Display the frame
    cv2.imshow("Combined YOLO Detection & Segmentation (GPU)", frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
