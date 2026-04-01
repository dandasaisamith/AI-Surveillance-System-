import cv2
import numpy as np
import time
from utils import YOLO, YOLO_AVAILABLE, DEVICE, torch
from tracking import PersonTracker, build_detections
from ai_analysis import analyze_person_attributes

class SurveillancePipeline:
    def __init__(self):
        self.model = None
        self.tracker = PersonTracker()
        self.frame_count = 0
        self.process_every_n = 2
        self.target_size = (640, 480)
        
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO("yolov8n.pt")
                if DEVICE == "cuda" and torch:
                    self.model.to(DEVICE)
            except:
                self.model = None
    
    def process_frame(self, frame):
        """Process single frame"""
        self.frame_count += 1
        
        # Resize for performance
        frame = cv2.resize(frame, self.target_size)
        
        # Skip frames for FPS optimization
        if self.frame_count % self.process_every_n != 0:
            return frame, []
        
        if not self.model:
            return frame, []
        
        try:
            # Run YOLO detection
            results = self.model(
                frame,
                verbose=False,
                conf=0.4,
                classes=[0],  # person only
                device=DEVICE
            )
            
            # Build detections for tracker
            detections = build_detections(results[0])
            
            # Update tracker
            tracks = self.tracker.update(detections, frame)
            
            # Draw on frame
            annotated = results[0].plot()
            
            # Draw track IDs
            for track in tracks:
                x1, y1, x2, y2 = track["bbox"]
                track_id = track["id"]
                
                # Draw track ID
                cv2.putText(
                    annotated,
                    f"ID:{track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                
                # Check if idle
                if self.tracker.is_idle(track_id):
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(
                        annotated,
                        "IDLE",
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )
            
            return annotated, tracks
        except:
            return frame, []
    
    def get_person_crop(self, frame, bbox):
        """Extract person crop for analysis"""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return frame[y1:y2, x1:x2]
    
    def analyze_suspicious(self, tracks, zone_rect=None):
        """Determine suspicious tracks"""
        suspicious = []
        
        # Don't flag single person as suspicious
        if len(tracks) <= 1:
            return suspicious
        
        for track in tracks:
            track_id = track["id"]
            is_suspicious = False
            reason = []
            
            # Check idle
            if self.tracker.is_idle(track_id):
                is_suspicious = True
                reason.append("Prolonged idle behavior")
            
            # Check restricted zone
            if zone_rect and self.tracker.in_zone(track_id, zone_rect):
                is_suspicious = True
                reason.append("In restricted zone")
            
            if is_suspicious:
                suspicious.append({
                    "track_id": track_id,
                    "bbox": track["bbox"],
                    "reason": ", ".join(reason)
                })
        
        return suspicious
