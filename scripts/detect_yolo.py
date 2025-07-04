import cv2
from ultralytics import YOLO

def detect_yolo(video_path, output_path, model_path='yolo11n.pt'):
    """Detect vehicles using YOLOv11 and save annotated video."""
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    detections = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, classes=[2])  # Class 2: car
        frame_detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                frame_detections.append(([x1, y1, x2, y2], conf, 'car'))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Car: {conf:.2f}', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        detections.append(frame_detections)
        out.write(frame)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return detections