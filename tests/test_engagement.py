"""
Unit tests for Multimodal Engagement Fusion and Interval Analytics.
"""

import pytest
from app.engagement.engine import MultimodalEngagementEngine
from app.vision.tracker import ParticipantTrack

def test_instantaneous_engagement_bounds():
    engine = MultimodalEngagementEngine()
    
    # High signals
    high_score = engine.compute_instantaneous_engagement(
        attention_signal=95.0,
        movement_score=35.0,
        audio_active=True,
        presence_continuity=90.0
    )
    assert 0.0 <= high_score <= 100.0
    assert high_score > 70.0
    
    # Low / lethargic signals
    low_score = engine.compute_instantaneous_engagement(
        attention_signal=20.0,
        movement_score=2.0,  # Extreme stillness
        audio_active=False,
        presence_continuity=30.0
    )
    assert 0.0 <= low_score <= 100.0
    assert low_score < 50.0

def test_interval_partitioning_and_flags():
    engine = MultimodalEngagementEngine(interval_sec=10.0)
    
    tracks = {
        "P1": ParticipantTrack("P1", (100, 100), (0,0,10,10), 0.0, 30.0),
        "P2": ParticipantTrack("P2", (200, 200), (0,0,10,10), 0.0, 30.0)
    }
    
    # Interval 0: High attention & active audio
    # Interval 1: Sharp attention drop
    timeline_visual = [
        {"timestamp_sec": 2.0, "avg_attention_signal": 85.0, "avg_movement_score": 30.0, "tracked_ids": ["P1", "P2"]},
        {"timestamp_sec": 5.0, "avg_attention_signal": 85.0, "avg_movement_score": 30.0, "tracked_ids": ["P1", "P2"]},
        {"timestamp_sec": 12.0, "avg_attention_signal": 25.0, "avg_movement_score": 10.0, "tracked_ids": ["P1", "P2"]},
        {"timestamp_sec": 15.0, "avg_attention_signal": 20.0, "avg_movement_score": 5.0, "tracked_ids": ["P1", "P2"]}
    ]
    
    audio_profile = {
        "timeline_audio_records": [
            {"timestamp_sec": 2.0, "is_active": True},
            {"timestamp_sec": 5.0, "is_active": True},
            {"timestamp_sec": 12.0, "is_active": False},
            {"timestamp_sec": 15.0, "is_active": False}
        ]
    }
    
    intervals, summary = engine.analyze_session_intervals(
        session_duration_sec=20.0,
        tracks=tracks,
        timeline_visual=timeline_visual,
        audio_profile=audio_profile
    )
    
    assert len(intervals) == 2
    assert summary["total_intervals"] == 2
    # Interval 1 should trigger an attention drop flag
    assert intervals[1]["attention_drop_flag"] is True
    assert len(summary["attention_drops"]) >= 1
