"""
Video Processing Pipeline.
Extracts frames, coordinates face/head detection and tracking, calculates sustained presence,
and compiles per-frame visual indicators.
"""

import os
from typing import Dict, List, Any, Callable, Optional
import cv2
import numpy as np
from app.config import config
from app.vision.detector import ClassroomVisionDetector
from app.vision.tracker import ParticipantTracker, ParticipantTrack

class VideoProcessor:
    """
    Coordinates frame-level video analysis for attendance and engagement tracking.
    Supports file-based video processing and live frame callbacks.
    """

    def __init__(self, sample_fps: float = 2.0):
        self.sample_fps = sample_fps
        self.detector = ClassroomVisionDetector()
        self.tracker = ParticipantTracker(
            max_disappeared=config.tracker_max_disappeared_frames,
            distance_threshold=config.tracker_distance_threshold
        )

    def process_video_file(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Processes an input video file end-to-end.
        Extracts temporal indicators, tracks participants, and returns structured results.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / native_fps if native_fps > 0 else 0.0

        # Determine frame skip interval to achieve target sample_fps
        frame_interval = max(1, int(round(native_fps / self.sample_fps)))

        frame_idx = 0
        processed_count = 0
        timeline_visual_records: List[Dict[str, Any]] = []
        preview_frames: List[np.ndarray] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process only sampled frames
            if frame_idx % frame_interval == 0:
                current_time_sec = frame_idx / native_fps
                detections = self.detector.detect_in_frame(frame)
                tracks = self.tracker.update(detections, current_time_sec)

                # Compute frame-level summary indicators
                active_tracks = [t for t in tracks.values() if t.consecutive_missing_frames <= 2]
                active_count = len(active_tracks)

                if active_count > 0:
                    avg_att = sum(t.attention_scores[-1] for t in active_tracks) / active_count
                    avg_mot = sum(t.motion_scores[-1] for t in active_tracks) / active_count
                else:
                    avg_att = 0.0
                    avg_mot = 0.0

                record = {
                    "timestamp_sec": current_time_sec,
                    "active_participants": active_count,
                    "avg_attention_signal": round(avg_att, 2),
                    "avg_movement_score": round(avg_mot, 2),
                    "tracked_ids": [t.track_id for t in active_tracks]
                }
                timeline_visual_records.append(record)

                # Store occasional annotated preview frames (up to 12 frames across video)
                if len(preview_frames) < 12 and (processed_count % max(1, (total_frames // frame_interval // 10)) == 0):
                    annotated = self.detector.draw_annotations(frame, tracks, current_time_sec)
                    preview_frames.append(annotated)

                processed_count += 1

                if progress_callback and total_frames > 0:
                    pct = min(1.0, frame_idx / total_frames)
                    progress_callback(pct, f"Analyzing video frames... ({int(pct*100)}%)")

            frame_idx += 1

        cap.release()

        return {
            "duration_sec": duration_sec,
            "total_frames": total_frames,
            "processed_frames": processed_count,
            "tracks": self.tracker.tracks,
            "timeline_visual_records": timeline_visual_records,
            "preview_frames": preview_frames
        }

    def process_single_frame(
        self,
        frame: np.ndarray,
        current_time_sec: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Used for real-time webcam processing.
        Processes a single live frame, updates tracker, and returns annotated frame with metrics.
        """
        detections = self.detector.detect_in_frame(frame)
        tracks = self.tracker.update(detections, current_time_sec)

        active_tracks = [t for t in tracks.values() if t.consecutive_missing_frames <= 2]
        active_count = len(active_tracks)

        if active_count > 0:
            avg_att = sum(t.attention_scores[-1] for t in active_tracks) / active_count
            avg_mot = sum(t.motion_scores[-1] for t in active_tracks) / active_count
        else:
            avg_att = 0.0
            avg_mot = 0.0

        annotated = self.detector.draw_annotations(frame, tracks, current_time_sec)

        metrics = {
            "timestamp_sec": current_time_sec,
            "active_participants": active_count,
            "avg_attention_signal": round(avg_att, 2),
            "avg_movement_score": round(avg_mot, 2),
            "tracks": tracks
        }
        return annotated, metrics
