import cv2
import numpy as np
from ultralytics import YOLO
import torch

class SingleYOLODetector:
    def __init__(self, model_path, tile_size=(640, 640), overlap=0.2, conf_threshold=0.5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = YOLO(model_path).to(self.device)
        self.tile_size = tile_size
        self.overlap = overlap
        self.conf_threshold = conf_threshold  # Confidence threshold

    def get_tiles(self, image):
        height, width = image.shape[:2]
        tile_height, tile_width = self.tile_size
        stride_h, stride_w = int(tile_height * (1 - self.overlap)), int(tile_width * (1 - self.overlap))
        tiles, coords = [], []

        for y in range(0, height - tile_height + stride_h, stride_h):
            for x in range(0, width - tile_width + stride_w, stride_w):
                x1, y1 = min(x, width - tile_width), min(y, height - tile_height)
                tiles.append(image[y1:y1 + tile_height, x1:x1 + tile_width])
                coords.append((x1, y1))

        return tiles, coords

    def process_tile(self, tile, coord_offset):
        results = self.model(tile, device=self.device)
        detections = []

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confs, classes):
                if int(cls) == 0 and conf >= self.conf_threshold:  # Filter only people (class 0)
                    x1, y1, x2, y2 = box + np.array([*coord_offset, *coord_offset])
                    detections.append({'bbox': [x1, y1, x2, y2], 'conf': conf})

        return detections

    def detect(self, frame):
        tiles, coords = self.get_tiles(frame)
        all_detections = [self.process_tile(tile, coord) for tile, coord in zip(tiles, coords)]
        return [det for sublist in all_detections for det in sublist]

def draw_results(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, f"{det['conf']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return frame

def main():
    detector = SingleYOLODetector(model_path="../models/yolov8n.pt", conf_threshold=0.5)  # Confidence threshold set to 0.5
    cap = cv2.VideoCapture(r"E:\RoboCam\RoboCam\assets\dataset\video-dataset\market-square.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        frame = draw_results(frame, detections)

        cv2.imshow("Person Detection (YOLOv8)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
