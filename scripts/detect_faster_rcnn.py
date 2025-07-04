# Detect vehicles using Faster R-CNN (via Detectron2) for comparison with YOLOv11
# Use Detectron2's pre-trained faster R-CNN model, process video frames, output bounding boxes and save annotated video

import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo

def detect_faster_rcnn(video_path, output_path):
    """Detect vehicles using Faster R-CNN and save annotated video."""
    # Configure model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)
    
    # Open video
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
        outputs = predictor(frame)
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        
        frame_detections = []
        for box, score, cls in zip(boxes, scores, classes):
            if cls == 2:  # Class 2: car
                x1, y1, x2, y2 = map(int, box)
                frame_detections.append(([x1, y1, x2, y2], score, 'car'))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Car: {score:.2f}', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        detections.append(frame_detections)
        out.write(frame)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return detections