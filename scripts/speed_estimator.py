import numpy as np

class SpeedEstimator:
    """Estimate vehicle speeds from tracked centroids."""
    def __init__(self, fps, pixel_to_meter=0.1, speed_limit=120):
        self.fps = fps
        self.pixel_to_meter = pixel_to_meter
        self.speed_limit = speed_limit
        self.tracks = {}
    
    def estimate_speed(self, track_id, bbox, timestamp):
        """Estimate speed for a track."""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        if track_id in self.tracks:
            prev_x, prev_y, prev_time = self.tracks[track_id]
            pixel_distance = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
            speed_ms = pixel_distance * self.pixel_to_meter * self.fps
            speed_kmh = speed_ms * 3.6
        else:
            speed_kmh = 0
        
        self.tracks[track_id] = (center_x, center_y, timestamp)
        return speed_kmh
    
    def is_speeding(self, speed_kmh):
        """Check if vehicle exceeds speed limit."""
        return speed_kmh > self.speed_limit