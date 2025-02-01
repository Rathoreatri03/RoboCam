import cv2
import numpy as np
from ultralytics import YOLO
import torch


def compute_iou(box1, box2):
    """Calculate IOU between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


class CombinedTiledDetector:
    def __init__(self, detection_model_path, segmentation_model_path, tile_size=(640, 640), overlap=0.2):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load both models
        self.detection_model = YOLO(detection_model_path).to(self.device)
        self.segmentation_model = YOLO(segmentation_model_path).to(self.device)

        # Tiling parameters
        self.tile_size = tile_size
        self.overlap = overlap

    def get_tiles(self, image):
        """Split image into overlapping tiles."""
        height, width = image.shape[:2]
        tile_height, tile_width = self.tile_size

        stride_h = int(tile_height * (1 - self.overlap))
        stride_w = int(tile_width * (1 - self.overlap))

        tiles = []
        coords = []

        for y in range(0, height - tile_height + stride_h, stride_h):
            for x in range(0, width - tile_width + stride_w, stride_w):
                x1 = min(x, width - tile_width)
                y1 = min(y, height - tile_height)

                tile = image[y1:y1 + tile_height, x1:x1 + tile_width]
                tiles.append(tile)
                coords.append((x1, y1))

        return tiles, coords

    def process_tile(self, tile, coord_offset):
        """Process a single tile with both models."""
        # Get results from both models
        det_results = self.detection_model(tile, device=self.device)
        seg_results = self.segmentation_model(tile, device=self.device)

        detections = []
        segmentations = []

        # Process detection results
        if len(det_results) > 0 and det_results[0].boxes is not None:
            det_boxes = det_results[0].boxes.xyxy.cpu().numpy()
            det_classes = det_results[0].boxes.cls.cpu().numpy()
            det_confs = det_results[0].boxes.conf.cpu().numpy()

            for box, cls, conf in zip(det_boxes, det_classes, det_confs):
                if int(cls) == 0:  # person class
                    x1, y1, x2, y2 = box
                    # Adjust coordinates
                    x1 += coord_offset[0]
                    x2 += coord_offset[0]
                    y1 += coord_offset[1]
                    y2 += coord_offset[1]
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'conf': conf
                    })

        # Process segmentation results
        if len(seg_results) > 0 and seg_results[0].boxes is not None:
            seg_boxes = seg_results[0].boxes.xyxy.cpu().numpy()
            seg_classes = seg_results[0].boxes.cls.cpu().numpy()
            seg_confs = seg_results[0].boxes.conf.cpu().numpy()

            for box, cls, conf in zip(seg_boxes, seg_classes, seg_confs):
                if int(cls) == 0:  # person class
                    x1, y1, x2, y2 = box
                    # Adjust coordinates
                    x1 += coord_offset[0]
                    x2 += coord_offset[0]
                    y1 += coord_offset[1]
                    y2 += coord_offset[1]
                    segmentations.append({
                        'bbox': [x1, y1, x2, y2],
                        'conf': conf
                    })

        return detections, segmentations

    def detect(self, frame):
        """Perform detection on tiled image using both models."""
        tiles, coords = self.get_tiles(frame)
        all_detections = []
        all_segmentations = []

        for tile, coord in zip(tiles, coords):
            detections, segmentations = self.process_tile(tile, coord)
            all_detections.extend(detections)
            all_segmentations.extend(segmentations)

        return all_detections, all_segmentations


def find_matching_boxes(detections, segmentations, iou_threshold=0.5):
    """Find matching boxes between detection and segmentation results."""
    matching_pairs = []

    for det in detections:
        det_box = det['bbox']
        for seg in segmentations:
            seg_box = seg['bbox']
            iou = compute_iou(det_box, seg_box)

            if iou > iou_threshold:
                # Calculate combined box
                x1 = min(det_box[0], seg_box[0])
                y1 = min(det_box[1], seg_box[1])
                x2 = max(det_box[2], seg_box[2])
                y2 = max(det_box[3], seg_box[3])

                avg_conf = (det['conf'] + seg['conf']) / 2
                matching_pairs.append({
                    'bbox': [x1, y1, x2, y2],
                    'conf': avg_conf,
                    'iou': iou
                })

    return matching_pairs


def draw_results(frame, detections, segmentations, matching_pairs):
    """Draw detection boxes, segmentation boxes, and overlapping boxes."""
    # Draw detection boxes in green
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.putText(frame, f"D:{det['conf']:.2f}", (x1, y1 - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw segmentation boxes in blue
    for seg in segmentations:
        x1, y1, x2, y2 = map(int, seg['bbox'])
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # cv2.putText(frame, f"S:{seg['conf']:.2f}", (x1, y1 - 25),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Draw matching boxes in yellow
    for match in matching_pairs:
        x1, y1, x2, y2 = map(int, match['bbox'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, f"M:{match['conf']:.2f} IOU:{match['iou']:.2f}",
                    (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return frame


def main():
    # Initialize detector with both models
    detector = CombinedTiledDetector(
        detection_model_path="../models/yolov8n.pt",
        segmentation_model_path="../models/yolov8n-seg.pt",
        tile_size=(640, 640),
        overlap=0.2
    )

    # Load video
    video_path = r"E:\RoboCam\RoboCam\assets\dataset\video-dataset\market-square.mp4"
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get detections from both models
        detections, segmentations = detector.detect(frame)

        # Find matching boxes
        matching_pairs = find_matching_boxes(detections, segmentations, iou_threshold=0.5)

        # Draw results
        frame = draw_results(frame, detections, segmentations, matching_pairs)

        # Display frame
        cv2.imshow("Combined YOLOv8 Detection & Segmentation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()