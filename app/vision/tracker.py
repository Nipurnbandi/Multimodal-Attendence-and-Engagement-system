"""
Multi-Participant Tracking and Sustained Presence Engine.
Tracks detected participants across video frames using centroid distance and bounding-box matching.
Eliminates single-frame false positives via sustained presence thresholds.
"""

import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from app.config import config

@dataclass
class ParticipantTrack:
    """Maintains continuous tracking history and metrics for an individual participant."""
    track_id: str
    centroid: Tuple[int, int]
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    first_seen_sec: float
    last_seen_sec: float
    detection_timestamps: List[float] = field(default_factory=list)
    motion_scores: List[float] = field(default_factory=list)
    attention_scores: List[float] = field(default_factory=list)
    consecutive_missing_frames: int = 0
    total_detections: int = 0

    @property
    def total_detected_sec(self) -> float:
        """Estimate total active presence duration based on detection count and sampling interval."""
        if not self.detection_timestamps:
            return 0.0
        # Time span of detections
        return max(0.0, self.last_seen_sec - self.first_seen_sec)

    def calculate_presence_ratio(self, total_session_sec: float) -> float:
        """Ratio of session duration that the participant was actively present."""
        if total_session_sec <= 0:
            return 0.0
        # If participant was present throughout, ratio approaches 1.0
        active_duration = len(self.detection_timestamps) / max(1.0, config.frame_sample_rate_fps)
        return min(1.0, active_duration / total_session_sec)

    def get_sustained_status(self, total_session_sec: float) -> Tuple[str, str]:
        """
        Determine attendance status based on sustained presence criteria.
        Returns (status, explanation)
        """
        # Active duration based on sampled frames
        active_duration = len(self.detection_timestamps) / max(1.0, config.frame_sample_rate_fps)
        ratio = active_duration / max(1.0, total_session_sec)

        # Minimum required duration (e.g. 10 seconds or 35% of session length for short demos)
        min_sec = min(config.sustained_presence_threshold_sec, total_session_sec * 0.3)

        if active_duration >= min_sec and ratio >= config.presence_ratio_threshold:
            return "Present", f"Sustained presence detected ({active_duration:.1f}s, {ratio*100:.1f}% of session)"
        elif active_duration < min_sec and active_duration > 0:
            return "Transient", f"Temporary appearance only ({active_duration:.1f}s) - filtered to prevent false positive"
        else:
            return "Absent", "Insufficient detection across session"


class ParticipantTracker:
    """
    Centroid and proximity-based tracker for classroom participants.
    Assigns consistent participant IDs and logs temporal indicators.
    """

    def __init__(self, max_disappeared: int = 15, distance_threshold: float = 90.0):
        self.next_id = 1
        self.tracks: Dict[str, ParticipantTrack] = {}
        self.max_disappeared = max_disappeared
        self.distance_threshold = distance_threshold

    def _calculate_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def register_new_track(self, centroid: Tuple[int, int], bbox: Tuple[int, int, int, int], timestamp_sec: float) -> str:
        track_id = f"Participant_{self.next_id:02d}"
        self.next_id += 1
        track = ParticipantTrack(
            track_id=track_id,
            centroid=centroid,
            bbox=bbox,
            first_seen_sec=timestamp_sec,
            last_seen_sec=timestamp_sec,
            detection_timestamps=[timestamp_sec],
            total_detections=1
        )
        self.tracks[track_id] = track
        return track_id

    def update(
        self,
        detections: List[Dict],
        timestamp_sec: float
    ) -> Dict[str, ParticipantTrack]:
        """
        Update tracker with new frame detections.
        Each detection is a dict with {'bbox': (x, y, w, h), 'attention': float, 'motion': float}
        """
        # If no tracks exist yet, register all detections
        if not self.tracks:
            for det in detections:
                bbox = det["bbox"]
                cx = bbox[0] + bbox[2] // 2
                cy = bbox[1] + bbox[3] // 2
                tid = self.register_new_track((cx, cy), bbox, timestamp_sec)
                self.tracks[tid].attention_scores.append(det.get("attention", 70.0))
                self.tracks[tid].motion_scores.append(det.get("motion", 30.0))
            return self.tracks

        # If no detections in current frame, increment missing count
        if not detections:
            for track in self.tracks.values():
                track.consecutive_missing_frames += 1
            return self.tracks

        # Match detections to existing tracks via nearest centroid
        matched_tracks = set()
        matched_detections = set()

        # Compute cost/distance matrix
        track_keys = list(self.tracks.keys())
        distances = []

        for det_idx, det in enumerate(detections):
            bbox = det["bbox"]
            dcx = bbox[0] + bbox[2] // 2
            dcy = bbox[1] + bbox[3] // 2
            for tid in track_keys:
                tcx, tcy = self.tracks[tid].centroid
                dist = self._calculate_distance((dcx, dcy), (tcx, tcy))
                distances.append((dist, det_idx, tid))

        # Sort matches by shortest distance
        distances.sort(key=lambda x: x[0])

        for dist, det_idx, tid in distances:
            if det_idx in matched_detections or tid in matched_tracks:
                continue
            if dist < self.distance_threshold:
                # Associate detection with track
                matched_detections.add(det_idx)
                matched_tracks.add(tid)
                det = detections[det_idx]
                bbox = det["bbox"]
                cx = bbox[0] + bbox[2] // 2
                cy = bbox[1] + bbox[3] // 2

                track = self.tracks[tid]
                track.centroid = (cx, cy)
                track.bbox = bbox
                track.last_seen_sec = timestamp_sec
                track.detection_timestamps.append(timestamp_sec)
                track.consecutive_missing_frames = 0
                track.total_detections += 1
                track.attention_scores.append(det.get("attention", 70.0))
                track.motion_scores.append(det.get("motion", 30.0))

        # Register unmatched detections as new participants
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_detections:
                bbox = det["bbox"]
                cx = bbox[0] + bbox[2] // 2
                cy = bbox[1] + bbox[3] // 2
                tid = self.register_new_track((cx, cy), bbox, timestamp_sec)
                self.tracks[tid].attention_scores.append(det.get("attention", 70.0))
                self.tracks[tid].motion_scores.append(det.get("motion", 30.0))

        # Increment missing count for tracks that were not matched
        for tid, track in self.tracks.items():
            if tid not in matched_tracks:
                track.consecutive_missing_frames += 1

        return self.tracks
