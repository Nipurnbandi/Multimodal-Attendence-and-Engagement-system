"""
Unit tests for Computer Vision & Spatial Tracking Modules.
"""

import numpy as np
import pytest
from app.vision.tracker import ParticipantTracker, ParticipantTrack
from app.vision.detector import ClassroomVisionDetector

def test_tracker_registration():
    tracker = ParticipantTracker(max_disappeared=10, distance_threshold=80.0)
    detections = [
        {"bbox": (100, 100, 50, 50), "attention": 85.0, "motion": 30.0},
        {"bbox": (300, 100, 50, 50), "attention": 75.0, "motion": 20.0}
    ]
    tracks = tracker.update(detections, timestamp_sec=0.5)
    assert len(tracks) == 2
    assert "Participant_01" in tracks
    assert "Participant_02" in tracks
    assert tracks["Participant_01"].first_seen_sec == 0.5
    assert tracks["Participant_01"].consecutive_missing_frames == 0

def test_tracker_association():
    tracker = ParticipantTracker(max_disappeared=10, distance_threshold=80.0)
    # Frame 1
    tracker.update([{"bbox": (100, 100, 50, 50), "attention": 80.0, "motion": 25.0}], timestamp_sec=0.0)
    # Frame 2: slightly shifted (104, 102)
    tracks = tracker.update([{"bbox": (104, 102, 50, 50), "attention": 82.0, "motion": 28.0}], timestamp_sec=0.5)
    
    assert len(tracks) == 1
    assert "Participant_01" in tracks
    assert tracks["Participant_01"].total_detections == 2
    assert tracks["Participant_01"].last_seen_sec == 0.5

def test_sustained_presence_evaluation():
    # 1. Transient participant: seen only for 2 detections (1.0 sec active in a 60s session)
    transient_track = ParticipantTrack(
        track_id="P_Transient",
        centroid=(100, 100),
        bbox=(80, 80, 40, 40),
        first_seen_sec=10.0,
        last_seen_sec=11.0,
        detection_timestamps=[10.0, 10.5],
        total_detections=2
    )
    status, notes = transient_track.get_sustained_status(total_session_sec=60.0)
    assert status == "Transient"
    assert "filtered" in notes.lower()

    # 2. Sustained participant: actively detected across 40 frames (~20s in a 40s session)
    sustained_track = ParticipantTrack(
        track_id="P_Sustained",
        centroid=(200, 200),
        bbox=(180, 180, 40, 40),
        first_seen_sec=2.0,
        last_seen_sec=38.0,
        detection_timestamps=[i * 0.5 for i in range(40)],
        total_detections=40
    )
    status2, notes2 = sustained_track.get_sustained_status(total_session_sec=40.0)
    assert status2 == "Present"
    assert "sustained" in notes2.lower()

def test_vision_detector_empty_frame():
    detector = ClassroomVisionDetector()
    empty_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = detector.detect_in_frame(empty_frame)
    assert isinstance(detections, list)
    assert len(detections) == 0
