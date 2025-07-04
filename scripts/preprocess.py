# Purpose of the file - extract video metadata( fps, resolution)
# and optionally preprocess frames(resize, convert formats)
# use opencv to read mp4 video, extract FPS, w, h for speed estimation adn video processing

import cv2
import os

def get_video_info(video_path):
    """Extract FPS, width and height from a video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, width, height

def extract_clip(video_path, output_path, start_sec=0, duration_sec=10):
    """Extract a short clip from the video for testing"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Fixed typo: VideoWriter_fourcc (not VideoWrtiter_fourcc)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (width, height))
    
    start_frame = int(start_sec * fps)
    end_frame = int((start_sec + duration_sec) * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for _ in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    cap.release()
    out.release()
    return output_path