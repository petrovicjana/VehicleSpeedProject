# test.py

# test.py
import cv2
import time
import os
from detectors import YOLODetector
from trackers import SORTTracker

def test_detector_tracker(detector, tracker, detector_name, tracker_name, video_path, output_path=None, max_frames=100):
    print(f"\n{'='*60}")
    print(f"Testing {detector_name} + {tracker_name}")
    print(f"{'='*60}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    # Initialize video writer
    out = None
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving output to: {os.path.abspath(output_path)}")
    
    frame_count = 0
    detection_count = 0
    total_time = 0
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_count+1}")
            break
        
        # Detect objects
        start_time = time.time()
        detections = detector.detect(frame)
        # Track objects
        tracks = tracker.update(detections)
        elapsed_time = time.time() - start_time
        
        total_time += elapsed_time
        detection_count += len(detections)
        
        # Draw tracks
        for bbox, score, track_id in tracks:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID {track_id}: {score:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add info text
        info_text = f"{detector_name}+{tracker_name} | Frame: {frame_count+1} | Tracks: {len(tracks)} | FPS: {1/elapsed_time:.1f}"
        cv2.putText(frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if out:
            out.write(frame)
        
        frame_count += 1
        
        if frame_count % 10 == 0:
            print(f"Processed {frame_count}/{min(max_frames, total_frames)} frames | "
                  f"Avg FPS: {frame_count/total_time:.1f} | "
                  f"Tracks: {detection_count}")
    
    # Cleanup
    cap.release()
    if out:
        out.release()
        if os.path.exists(output_path):
            print(f"Output file exists at: {os.path.abspath(output_path)}")
    
    # Summary
    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_tracks = detection_count / frame_count if frame_count > 0 else 0
    
    print(f"\nResults:")
    print(f"  Frames processed: {frame_count}")
    print(f"  Total tracks: {detection_count}")
    print(f"  Avg tracks/frame: {avg_tracks:.2f}")
    print(f"  Avg FPS: {avg_fps:.2f}")
    print(f"  Total time: {total_time:.2f}s")
    if output_path:
        print(f"  Output saved to: {output_path}")

def main():
    VIDEO_PATH = "test_clip.mp4"  # Update with your video path
    MAX_FRAMES = 100
    
    print("Starting YOLOv8 + SORT test...")
    print(f"Input video: {VIDEO_PATH}")
    
    try:
        print("\nTesting YOLOv8 + SORT...")
        yolo_detector = YOLODetector()
        sort_tracker = SORTTracker(max_age=1, min_hits=3, iou_threshold=0.3)
        test_detector_tracker(yolo_detector, sort_tracker, "YOLOv8", "SORT", VIDEO_PATH, 
                             output_path="outputs/output_yolo_sort.mp4", max_frames=MAX_FRAMES)
    except Exception as e:
        print(f"YOLOv8 + SORT test failed: {e}")

if __name__ == "__main__":
    main()



""" import cv2
import time
from detectors import YOLODetector, FasterRCNNDetector, SSDDetector

def test_detector(detector, detector_name, video_path, output_path=None, max_frames=100):
    print(f"\n{'='*60}")
    print(f"Testing {detector_name}")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    detection_count = 0
    total_time = 0
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        detections = detector.detect(frame)
        elapsed_time = time.time() - start_time
        
        total_time += elapsed_time
        detection_count += len(detections)
        
        # Draw detections
        for bbox, score in detections:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Car: {score:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        info_text = f"{detector_name} | Frame: {frame_count+1} | Cars: {len(detections)} | FPS: {1/elapsed_time:.1f}"
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if out:
            out.write(frame)
        
        frame_count += 1
        
        if frame_count % 10 == 0:
            print(f"Processed {frame_count}/{min(max_frames, total_frames)} frames | "
                  f"Avg FPS: {frame_count/total_time:.1f} | "
                  f"Detections: {detection_count}")
    
    cap.release()
    if out:
        out.release()
    
    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_detections = detection_count / frame_count if frame_count > 0 else 0
    
    print(f"\nResults:")
    print(f"  Frames processed: {frame_count}")
    print(f"  Total detections: {detection_count}")
    print(f"  Avg detections/frame: {avg_detections:.2f}")
    print(f"  Avg FPS: {avg_fps:.2f}")
    print(f"  Total time: {total_time:.2f}s")
    if output_path:
        print(f"  Output saved to: {output_path}")

def main():
    VIDEO_PATH = "data/test_clip.mp4"  
    MAX_FRAMES = 100
    
    print("Starting detector tests...")
    print(f"Input video: {VIDEO_PATH}")
    
    try:
        print("\n[1/3] Testing YOLOv8...")
        yolo_detector = YOLODetector()
        test_detector(yolo_detector, "YOLOv8", VIDEO_PATH, 
                     output_path="output_yolo.mp4", max_frames=MAX_FRAMES)
    except Exception as e:
        print(f"YOLOv8 test failed: {e}")
    
    try:
        print("\n[2/3] Testing Faster R-CNN...")
        frcnn_detector = FasterRCNNDetector()
        test_detector(frcnn_detector, "Faster R-CNN", VIDEO_PATH,
                     output_path="output_frcnn.mp4", max_frames=MAX_FRAMES)
    except Exception as e:
        print(f"Faster R-CNN test failed: {e}")
    
    try:
        print("\n[3/3] Testing SSD...")
        ssd_detector = SSDDetector()
        test_detector(ssd_detector, "SSD", VIDEO_PATH,
                     output_path="output_ssd.mp4", max_frames=MAX_FRAMES)
    except Exception as e:
        print(f"SSD test failed: {e}")
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)

if __name__ == "__main__":
    main() """