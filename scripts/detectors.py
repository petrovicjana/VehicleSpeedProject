import cv2
import torch
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssd300_vgg16


class YOLODetector:
    """YOLOv8 - Fastest detector"""
    def __init__(self):
        self.model = YOLO('yolov8n.pt')  # Downloads automatically first time
    
    def detect(self, frame):
        # YOLO classes: 2=car, 3=motorcycle, 5=bus, 7=truck
        vehicle_classes = [2, 3, 5, 7]
        results = self.model(frame, classes=vehicle_classes, conf=0.5, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append(([x1, y1, x2, y2], conf))
        return detections


class FasterRCNNDetector:
    """Faster R-CNN - Most accurate but slowest"""
    def __init__(self):
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
    
    def detect(self, frame):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to tensor
        img_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        
        with torch.no_grad():
            pred = self.model(img_tensor)[0]
        
        detections = []
        for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
            if label == 3 and score >= 0.5:  # label 3 = car
                x1, y1, x2, y2 = map(int, box)
                detections.append(([x1, y1, x2, y2], float(score)))
        return detections


class SSDDetector:
    """SSD - Good balance of speed and accuracy"""
    def __init__(self):
        self.model = ssd300_vgg16(pretrained=True)
        self.model.eval()
    
    def detect(self, frame):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to tensor
        img_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        
        with torch.no_grad():
            pred = self.model(img_tensor)[0]
        
        detections = []
        for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
            if label == 3 and score >= 0.5:  # label 3 = car
                x1, y1, x2, y2 = map(int, box)
                detections.append(([x1, y1, x2, y2], float(score)))
        return detections