import cv2

class VideoProcessor:
    """Handle video reading, writing, and metadata extraction."""
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def get_metadata(self):
        """Return FPS, width, and height."""
        return self.fps, self.width, self.height
    
    def read_frame(self):
        """Read the next frame."""
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def create_writer(self, output_path):
        """Create a video writer."""
        return cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*'mp4v'), 
            self.fps, (self.width, self.height)
        )
    
    def extract_clip(self, output_path, start_sec=0, duration_sec=10):
        """Extract a short clip for testing."""
        writer = self.create_writer(output_path)
        start_frame = int(start_sec * self.fps)
        end_frame = int((start_sec + duration_sec) * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for _ in range(start_frame, end_frame):
            frame = self.read_frame()
            if frame is None:
                break
            writer.write(frame)
        
        writer.release()
        return output_path
    
    def release(self):
        """Release video capture."""
        self.cap.release()