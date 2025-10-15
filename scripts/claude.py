import cv2
import time
import os
import numpy as np
from detectors import YOLODetector
from trackers import ByteTrackTracker

def test_pipeline(detector, tracker, video_path, output_path, max_frames=None):
    print(f"\n{'='*60}")
    print(f"Testing YOLOv8 + ByteTrack")
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
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Use better codec - avc1 or H264
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Better compatibility than mp4v
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        cap.release()
        return
    
    frame_count = 0
    processed_count = 0
    track_count = 0
    total_time = 0
    max_frames = total_frames if max_frames is None else min(max_frames, total_frames)
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_count+1}")
            break
        
        frame_count += 1
        
        # Process every frame (remove frame skipping for smooth output)
        # If you want to skip frames, do it before writing to output
        
        start_time = time.time()
        detections = detector.detect(frame)
        
        if processed_count == 0:  # Print only for first frame to reduce clutter
            print(f"Frame {frame_count} detections: {len(detections)}")
        
        # Validate and filter detections
        valid_detections = []
        for bbox, conf in detections:
            x1, y1, x2, y2 = map(int, bbox)
            if 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height and conf >= 0.3:
                valid_detections.append((bbox, conf))
        
        if processed_count == 0:
            print(f"Valid detections: {len(valid_detections)}")
        
        tracks = tracker.update(valid_detections, frame)
        elapsed_time = time.time() - start_time
        
        total_time += elapsed_time
        track_count += len(tracks)
        processed_count += 1
        
        # Draw tracks on frame
        for bbox, score, track_id in tracks:
            x1, y1, x2, y2 = map(int, np.array(bbox).flatten())
            
            # Constrain to frame bounds
            x1 = max(0, min(x1, width - 1))
            x2 = max(x1 + 1, min(x2, width))
            y1 = max(0, min(y1, height - 1))
            y2 = max(y1 + 1, min(y2, height))
            
            box_width = x2 - x1
            box_height = y2 - y1
            
            # Skip unreasonably large boxes
            if box_width > width / 2 or box_height > height / 2:
                print(f"Warning: Large box at frame {frame_count}, ID {track_id}: {x1},{y1},{x2},{y2}")
                continue
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"ID {track_id}: {score:.2f}"
            cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display info on frame
        fps_display = f"{1/elapsed_time:.1f}" if elapsed_time > 0 else "0.0"
        info_text = f"YOLOv8+ByteTrack | Frame: {frame_count} | Tracks: {len(tracks)} | FPS: {fps_display}"
        cv2.putText(frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Write frame to output
        out.write(frame)
        
        # Progress update every 30 frames
        if frame_count % 30 == 0:
            avg_fps = processed_count / total_time if total_time > 0 else 0
            print(f"Processed {frame_count}/{max_frames} frames | Avg FPS: {avg_fps:.1f} | Total Tracks: {track_count}")
    
    # Release resources
    cap.release()
    out.release()
    
    # Verify output file
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"\n✅ Output file created successfully!")
        print(f"   Location: {os.path.abspath(output_path)}")
        print(f"   Size: {file_size:.2f} MB")
    else:
        print(f"\n❌ Warning: Output file not created: {output_path}")
    
    # Final statistics
    avg_fps = processed_count / total_time if total_time > 0 else 0
    avg_tracks = track_count / processed_count if processed_count > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Results Summary:")
    print(f"{'='*60}")
    print(f"  Frames processed: {processed_count}")
    print(f"  Total tracks detected: {track_count}")
    print(f"  Avg tracks per frame: {avg_tracks:.2f}")
    print(f"  Processing FPS: {avg_fps:.2f}")
    print(f"  Total processing time: {total_time:.2f}s")
    print(f"  Video FPS: {fps}")
    print(f"{'='*60}\n")

def main():
    VIDEO_PATH = "test_clip_30s.mp4"
    MAX_FRAMES = 200  # Set to None for full video
    
    print("Starting YOLOv8 + ByteTrack test...")
    print(f"Input video: {VIDEO_PATH}")
    
    # Verify input video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Error: Video file not found: {VIDEO_PATH}")
        return
    
    # Initialize detector and tracker
    print("Initializing YOLOv8 detector...")
    detector = YOLODetector()
    
    print("Initializing ByteTrack tracker...")
    tracker = ByteTrackTracker(track_thresh=0.3, match_thresh=0.7, frame_rate=30)
    
    output_path = "/content/drive/MyDrive/Colab Projects/VehicleSpeedProject/scripts/outputs/output_yolo_bytetrack.mp4"
    
    try:
        test_pipeline(detector, tracker, VIDEO_PATH, output_path, max_frames=MAX_FRAMES)
        print("✅ Test completed successfully!")
    except Exception as e:
        print(f"❌ YOLOv8 + ByteTrack test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()



















































""" import cv2
import time
import os
import numpy as np
from detectors import YOLODetector, FasterRCNNDetector, SSDDetector
from trackers import SORTTracker, DeepSortTracker, ByteTrackTracker

def test_pipeline(detector, tracker, detector_name, tracker_name, video_path, output_path, max_frames=None):
    print(f"\n{'='*60}")
    print(f"Testing {detector_name} + {tracker_name}")
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
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Try 'XVID' if mp4v fails
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        cap.release()
        return
    
    frame_count = 0
    track_count = 0
    total_time = 0
    max_frames = total_frames if max_frames is None else min(max_frames, total_frames)
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_count+1}")
            break
        
        # Skip every other frame to reduce motion blur
        if frame_count % 2 != 0:
            frame_count += 1
            continue
        
        start_time = time.time()
        detections = detector.detect(frame)
        if isinstance(tracker, (DeepSortTracker, ByteTrackTracker)):
            tracks = tracker.update(detections, frame)
        else:
            tracks = tracker.update(detections)
        elapsed_time = time.time() - start_time
        
        total_time += elapsed_time
        track_count += len(tracks)
        
        for bbox, score, track_id in tracks:
            x1, y1, x2, y2 = map(int, np.array(bbox).flatten())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID {track_id}: {score:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        fps_display = f"{1/elapsed_time:.1f}" if elapsed_time > 0 else "0.0"
        info_text = f"{detector_name}+{tracker_name} | Frame: {frame_count+1} | Tracks: {len(tracks)} | FPS: {fps_display}"
        cv2.putText(frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        out.write(frame)
        
        frame_count += 1
        
        if frame_count % 10 == 0:
            print(f"Processed {frame_count}/{max_frames} frames | Avg FPS: {frame_count/total_time:.1f if total_time > 0 else 0} | Tracks: {track_count}")
    
    cap.release()
    out.release()
    
    if os.path.exists(output_path):
        print(f"Output file exists at: {os.path.abspath(output_path)}")
    else:
        print(f"Warning: Output file not created: {output_path}")
    
    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_tracks = track_count / frame_count if frame_count > 0 else 0
    
    print(f"\nResults:")
    print(f"  Frames processed: {frame_count}")
    print(f"  Total tracks: {track_count}")
    print(f"  Avg tracks/frame: {avg_tracks:.2f}")
    print(f"  Avg FPS: {avg_fps:.2f}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Output saved to: {output_path}")

def main():
    VIDEO_PATH = "test_clip_30s.mp4"
    MAX_FRAMES = 200  # Set to None for full video
    
    print("Starting all pipeline tests...")
    print(f"Input video: {VIDEO_PATH}")
    
    detectors = {
        "YOLOv8": YOLODetector(),
        "FasterRCNN": FasterRCNNDetector(),
        "SSD": SSDDetector()
    }
    
    trackers = {
        "SORT": SORTTracker(max_age=50, min_hits=2, iou_threshold=0.15),
        "DeepSort": DeepSortTracker(max_age=50, nn_budget=200),
        "ByteTrack": ByteTrackTracker(track_thresh=0.3, match_thresh=0.7, frame_rate=30)
    }
    
    for det_name, detector in detectors.items():
        for trk_name, tracker in trackers.items():
            output_path = f"/content/drive/MyDrive/Colab Projects/VehicleSpeedProject/scripts/outputs/output_{det_name.lower()}_{trk_name.lower()}.mp4"
            try:
                test_pipeline(detector, tracker, det_name, trk_name, VIDEO_PATH, output_path, max_frames=MAX_FRAMES)
            except Exception as e:
                print(f"{det_name} + {trk_name} test failed: {e}")

if __name__ == "__main__":
    main() 


 """
 
""" import cv2
import time
import os
import numpy as np
from detectors import YOLODetector, FasterRCNNDetector, SSDDetector
from trackers import SORTTracker, DeepSortTracker, ByteTrackTracker


# --- Added slowdown preprocessing function ---
def slow_down_video(input_path, output_path, slow_factor=2.0):
    
    Creates a slowed-down copy of the video.
    Keeps all frames but reduces FPS so playback is slower.
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open input video: {input_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create output writer with reduced fps
    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps / slow_factor,
                          (width, height))

    print(f"⏳ Creating slowed video ({slow_factor}x slower)...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Slowed video saved to: {output_path}")
    return output_path
# --- End slowdown preprocessing function ---


def test_pipeline(detector, tracker, detector_name, tracker_name, video_path, output_path, max_frames=None):
    print(f"\n{'='*60}")
    print(f"Testing {detector_name} + {tracker_name}")
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

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        cap.release()
        return

    frame_count = 0
    track_count = 0
    total_time = 0
    max_frames = total_frames if max_frames is None else min(max_frames, total_frames)

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_count+1}")
            break

        # Skip every other frame to reduce motion blur
        if frame_count % 2 != 0:
            frame_count += 1
            continue

        start_time = time.time()
        detections = detector.detect(frame)
        if isinstance(tracker, (DeepSortTracker, ByteTrackTracker)):
            tracks = tracker.update(detections, frame)
        else:
            tracks = tracker.update(detections)
        elapsed_time = time.time() - start_time

        total_time += elapsed_time
        track_count += len(tracks)

        for bbox, score, track_id in tracks:
            x1, y1, x2, y2 = map(int, np.array(bbox).flatten())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID {track_id}: {score:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        fps_display = f"{1/elapsed_time:.1f}" if elapsed_time > 0 else "0.0"
        info_text = f"{detector_name}+{tracker_name} | Frame: {frame_count+1} | Tracks: {len(tracks)} | FPS: {fps_display}"
        cv2.putText(frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        out.write(frame)
        frame_count += 1

        if frame_count % 10 == 0:
            avg_fps = frame_count / total_time if total_time > 0 else 0
            print(f"Processed {frame_count}/{max_frames} frames | Avg FPS: {avg_fps:.1f} | Tracks: {track_count}")

    cap.release()
    out.release()

    if os.path.exists(output_path):
        print(f"Output file exists at: {os.path.abspath(output_path)}")
    else:
        print(f"Warning: Output file not created: {output_path}")

    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_tracks = track_count / frame_count if frame_count > 0 else 0

    print(f"\nResults:")
    print(f"  Frames processed: {frame_count}")
    print(f"  Total tracks: {track_count}")
    print(f"  Avg tracks/frame: {avg_tracks:.2f}")
    print(f"  Avg FPS: {avg_fps:.2f}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Output saved to: {output_path}")


def main():
    VIDEO_PATH = "test_clip.mp4"
    SLOWED_VIDEO_PATH = "test_clip_slow.mp4"  # --- Added path for slowed video ---
    MAX_FRAMES = 200  # Set to None for full video
    SLOW_FACTOR = 2.0  # --- Adjust slowdown here (2x, 3x, etc.) ---

    print("Starting all pipeline tests...")
    print(f"Input video: {VIDEO_PATH}")

    # --- Added slowdown preprocessing call ---
    if not os.path.exists(SLOWED_VIDEO_PATH):
        slow_down_video(VIDEO_PATH, SLOWED_VIDEO_PATH, slow_factor=SLOW_FACTOR)
    else:
        print(f"✅ Slowed video already exists: {SLOWED_VIDEO_PATH}")

    # Use slowed video for testing
    VIDEO_PATH = SLOWED_VIDEO_PATH
    # --- End slowdown preprocessing ---

    detectors = {
        "YOLOv8": YOLODetector(),
        "FasterRCNN": FasterRCNNDetector(),
        "SSD": SSDDetector()
    }

    trackers = {
        "SORT": SORTTracker(max_age=50, min_hits=2, iou_threshold=0.15),
        "DeepSort": DeepSortTracker(max_age=50, nn_budget=200),
        "ByteTrack": ByteTrackTracker(track_thresh=0.3, match_thresh=0.7, frame_rate=30)
    }

    for det_name, detector in detectors.items():
        for trk_name, tracker in trackers.items():
            output_path = f"/content/drive/MyDrive/Colab Projects/VehicleSpeedProject/scripts/outputs/output_{det_name.lower()}_{trk_name.lower()}.mp4"
            try:
                test_pipeline(detector, tracker, det_name, trk_name, VIDEO_PATH, output_path, max_frames=MAX_FRAMES)
            except Exception as e:
                print(f"{det_name} + {trk_name} test failed: {e}")


if __name__ == "__main__":
    main() """