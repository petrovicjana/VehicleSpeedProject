import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def detect_ssd(video_path, output_path, model_path='ssd_mobilenet_v2_coco_2018_03_29'):
    """Detect vehicles using SSD and save annotated video."""
    # Load model
    model = tf.saved_model.load(model_path)
    category_index = label_map_util.create_category_index_from_labelmap(
        'mscoco_label_map.pbtxt', use_display_name=True)
    
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
        image_np = np.array(frame)
        input_tensor = tf.convert_to_tensor([image_np])
        detections_dict = model(input_tensor)
        
        boxes = detections_dict['detection_boxes'][0].numpy()
        scores = detections_dict['detection_scores'][0].numpy()
        classes = detections_dict['detection_classes'][0].numpy().astype(np.int32)
        
        frame_detections = []
        for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
            if score > 0.5 and cls == 3:  # Class 3: car in COCO
                ymin, xmin, ymax, xmax = box
                x1, y1, x2, y2 = int(xmin * width), int(ymin * height), int(xmax * width), int(ymax * height)
                frame_detections.append(([x1, y1, x2, y2], score, 'car'))
                vis_util.draw_bounding_box_on_image_array(
                    image_np, ymin, xmin, ymax, xmax, display_str_list=[f'Car: {score:.2f}'], 
                    use_normalized_coordinates=True, color='green', thickness=2)
        detections.append(frame_detections)
        out.write(image_np)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return detections