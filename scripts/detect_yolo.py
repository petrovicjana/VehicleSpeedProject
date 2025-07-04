import cv2
from ultralytics import YOLO
import sys
from preprocess import get_video_info
sys.path.append('../scripts')

def detect_yolo(video_path, output_path, model_path='yolo11n.pt'):
    """Detect vehicles using YOLOv11 and save annotated video."""
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps, width, height = get_video_info(video_path)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Out object created using VideoWriter class is responsible for writitng the processed video frames to a new file
    # with bounding boxes drawn around detected vehicles
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    
    detections = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # to results we give a video and num of classes 
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
        # Each frame processed by YoloV11 with green bounding boxes and confidence scores is written to the
        # file using out.write, output video retans org video's properties : FPS, w, l
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return detections