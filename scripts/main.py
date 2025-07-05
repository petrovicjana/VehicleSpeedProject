import cv2
from scripts.utils import VideoProcessor
from scripts.detectors import YOLODetector, FasterRCNNDetector, SSDDetector
from scripts.trackers import DeepSORTTracker, ByteTrackTracker
from scripts.speed_estimator import SpeedEstimator

def process_video(video_path, output_path, detector_type='yolo', tracker_type='deepsort', speed_limit=120, pixel_to_meter=0.1):
    """Process video with detection, tracking, and speed estimation."""
    # Initialize components
    processor = VideoProcessor(video_path)
    fps, width, height = processor.get_metadata()
    
    if detector_type == 'yolo':
        detector = YOLODetector(model_path='/content/drive/My Drive/Colab Notebooks/VehicleSpeedProject/models/yolo11n.pt' if 'drive' in video_path else 'models/yolo11n.pt')
    elif detector_type == 'faster_rcnn':
        detector = FasterRCNNDetector()
    elif detector_type == 'ssd':
        detector = SSDDetector(model_path='/content/VehicleSpeedProject/models/ssd_mobilenet_v2_coco_2018_03_29/saved_model')
        detector.set_frame_size(width, height)
    else:
        raise ValueError(f"Unknown detector: {detector_type}")
    
    tracker = DeepSORTTracker() if tracker_type == 'deepsort' else ByteTrackTracker()
    speed_estimator = SpeedEstimator(fps, pixel_to_meter, speed_limit)
    writer = processor.create_writer(output_path)
    
    # Process video
    while True:
        frame = processor.read_frame()
        if frame is None:
            break
        
        # Detect objects
        detections = detector.detect(frame)
        
        # Track objects
        tracked_objects = tracker.update(detections, frame)
        
        # Annotate frame
        timestamp = processor.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        for track in tracked_objects:
            track_id = track['track_id'] if tracker_type == 'bytetrack' else track.track_id
            bbox = track['bbox'] if tracker_type == 'bytetrack' else track.to_tlbr()
            x1, y1, x2, y2 = map(int, bbox)
            
            speed_kmh = speed_estimator.estimate_speed(track_id, bbox, timestamp)
            color = (0, 0, 255) if speed_estimator.is_speeding(speed_kmh) else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'ID: {track_id} Speed: {speed_kmh:.1f} km/h', 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        writer.write(frame)
    
    processor.release()
    writer.release()
    cv2.destroyAllWindows()
    return output_path