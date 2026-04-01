from utils import DeepSort, DEEPSORT_AVAILABLE, DEVICE
import numpy as np

class PersonTracker:
    def __init__(self):
        self.tracker = None
        self.active_tracks = {}
        self.track_history = {}
        self.idle_threshold = 8.0
        self.movement_threshold = 20.0
        
        if DEEPSORT_AVAILABLE:
            try:
                self.tracker = DeepSort(
                    max_age=30,
                    n_init=3,
                    max_cosine_distance=0.3,
                    embedder="mobilenet",
                    embedder_gpu=(DEVICE == "cuda")
                )
            except:
                self.tracker = None
    
    def update(self, detections, frame):
        """Update tracks with new detections"""
        if not self.tracker or not detections:
            return []
        
        try:
            tracks = self.tracker.update_tracks(detections, frame=frame)
            active_tracks = []
            
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = int(track.track_id)
                ltrb = track.to_ltrb()
                
                if ltrb is None:
                    continue
                
                x1, y1, x2, y2 = map(int, ltrb)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                track_info = {
                    "id": track_id,
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy),
                    "class": "person",
                    "confidence": track.get_det_conf() or 0.0
                }
                
                active_tracks.append(track_info)
                self.active_tracks[track_id] = track_info
                
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                self.track_history[track_id].append((cx, cy))
                
                if len(self.track_history[track_id]) > 30:
                    self.track_history[track_id].pop(0)
            
            return active_tracks
        except:
            return []
    
    def is_idle(self, track_id):
        """Check if track is idle based on movement"""
        if track_id not in self.track_history:
            return False
        
        history = self.track_history[track_id]
        if len(history) < 10:
            return False
        
        recent = history[-10:]
        distances = []
        for i in range(1, len(recent)):
            dx = recent[i][0] - recent[i-1][0]
            dy = recent[i][1] - recent[i-1][1]
            distances.append(np.sqrt(dx*dx + dy*dy))
        
        avg_movement = np.mean(distances) if distances else 0
        return avg_movement < self.movement_threshold
    
    def in_zone(self, track_id, zone_rect):
        """Check if track is in restricted zone"""
        if track_id not in self.active_tracks or not zone_rect:
            return False
        
        cx, cy = self.active_tracks[track_id]["center"]
        x1, y1, x2, y2 = zone_rect
        return x1 <= cx <= x2 and y1 <= cy <= y2

def build_detections(yolo_result):
    """Convert YOLO detections to DeepSORT format"""
    detections = []
    
    if not yolo_result.boxes or len(yolo_result.boxes) == 0:
        return detections
    
    boxes = yolo_result.boxes.xyxy.cpu().numpy()
    confs = yolo_result.boxes.conf.cpu().numpy()
    classes = yolo_result.boxes.cls.cpu().numpy()
    
    for box, conf, cls in zip(boxes, confs, classes):
        if int(cls) == 0:  # person class
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], float(conf), 0))
    
    return detections
