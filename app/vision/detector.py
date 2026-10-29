"""
Vision Detector and Feature Extractor.
Extracts behavioral indicators from video frames:
- Face presence (frontal vs profile)
- Eye-region alertness
- Micro-movement / activity level via inter-frame pixel differencing
All signals are transparently defined as estimated indicators.
"""

import os
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from app.config import config, DATA_DIR

class ClassroomVisionDetector:
    """
    Lightweight, offline-capable vision detector for classroom sessions.
    Combines Haar cascades with frame differencing to estimate attention and movement.
    """

    def __init__(self):
        frontal_path = str(DATA_DIR / "haarcascade_frontalface_default.xml")
        profile_path = str(DATA_DIR / "haarcascade_profileface.xml")
        eye_path = str(DATA_DIR / "haarcascade_eye.xml")

        self.frontal_cascade = cv2.CascadeClassifier(frontal_path) if os.path.exists(frontal_path) else None
        self.profile_cascade = cv2.CascadeClassifier(profile_path) if os.path.exists(profile_path) else None
        self.eye_cascade = cv2.CascadeClassifier(eye_path) if os.path.exists(eye_path) else None

        self.prev_gray_frame: Optional[np.ndarray] = None

    def detect_in_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Processes a single BGR video frame.
        Returns a list of participant detections:
        [
            {
                'bbox': (x, y, w, h),
                'attention_signal': float (0-100),
                'movement_score': float (0-100),
                'orientation_type': 'frontal' | 'profile' | 'downward',
                'eyes_detected': int
            }
        ]
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Frontal face detection
        detections = []
        frontal_faces = ()
        if self.frontal_cascade and not self.frontal_cascade.empty():
            frontal_faces = self.frontal_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=2,
                minSize=(30, 30)
            )

        for (x, y, fw, fh) in frontal_faces:
            roi_gray = gray[y:y+fh, x:x+fw]
            eyes_count = 0
            if self.eye_cascade and not self.eye_cascade.empty():
                eyes = self.eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=2,
                    minSize=(10, 10)
                )
                eyes_count = len(eyes)

            # Heuristic estimated attention score (0-100)
            if eyes_count >= 2:
                attention_signal = 88.0
                orientation = "frontal_alert"
            elif eyes_count == 1:
                attention_signal = 78.0
                orientation = "frontal"
            else:
                # Frontal face without visible eyes may indicate looking down at desk/notebook
                attention_signal = 65.0
                orientation = "downward_desk"

            motion_score = self._compute_crop_movement(gray, (x, y, fw, fh))

            detections.append({
                "bbox": (int(x), int(y), int(fw), int(fh)),
                "attention_signal": float(attention_signal),
                "movement_score": float(motion_score),
                "orientation_type": orientation,
                "eyes_detected": int(eyes_count)
            })

        # 2. Profile face detection (if not overlapping an existing frontal detection)
        if self.profile_cascade and not self.profile_cascade.empty():
            profile_faces = self.profile_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=2,
                minSize=(30, 30)
            )
            for (px, py, pw, ph) in profile_faces:
                # Check for overlap with existing frontal detections
                overlapping = False
                for d in detections:
                    fx, fy, fw, fh = d["bbox"]
                    if (abs(px - fx) < fw * 0.6) and (abs(py - fy) < fh * 0.6):
                        overlapping = True
                        break

                if not overlapping:
                    motion_score = self._compute_crop_movement(gray, (px, py, pw, ph))
                    detections.append({
                        "bbox": (int(px), int(py), int(pw), int(ph)),
                        "attention_signal": 42.0,  # Turned sideways / side interaction indicator
                        "movement_score": float(motion_score),
                        "orientation_type": "profile_turned",
                        "eyes_detected": 0
                    })

        self.prev_gray_frame = gray.copy()
        return detections

    def _compute_crop_movement(self, current_gray: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """
        Calculates inter-frame pixel change within the participant's bounding box.
        Returns a normalized score between 0.0 and 100.0.
        """
        if self.prev_gray_frame is None:
            return 25.0  # Baseline movement

        x, y, w, h = bbox
        cur_crop = current_gray[y:y+h, x:x+w]
        prev_crop = self.prev_gray_frame[y:y+h, x:x+w]

        if cur_crop.shape != prev_crop.shape or cur_crop.size == 0:
            return 25.0

        # Absolute frame difference
        diff = cv2.absdiff(cur_crop, prev_crop)
        mean_diff = float(np.mean(diff))

        # Normalize: mean difference typically ranges from 0 to 40 in classroom video
        normalized = min(100.0, (mean_diff / 25.0) * 100.0)
        return round(normalized, 1)

    def draw_annotations(
        self,
        frame: np.ndarray,
        tracks: Dict[str, any],
        session_sec: float
    ) -> np.ndarray:
        """
        Renders bounding boxes, participant labels, and estimated indicator badges onto the frame.
        """
        annotated = frame.copy()

        for tid, track in tracks.items():
            if track.consecutive_missing_frames > 3:
                continue

            x, y, w, h = track.bbox
            status, _ = track.get_sustained_status(session_sec)

            # Color coding: Green for sustained present, Yellow for transient, Cyan default
            if status == "Present":
                color = (40, 200, 40)
            elif status == "Transient":
                color = (0, 215, 255)
            else:
                color = (200, 200, 200)

            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Draw participant label banner
            label = f"{tid} [{status}]"
            cv2.rectangle(annotated, (x, max(0, y - 22)), (x + w, y), color, -1)
            cv2.putText(
                annotated,
                label,
                (x + 4, max(14, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        return annotated
