"""
Unit tests for AI Teaching Suggestions Agent.
"""

import pytest
from app.agent.recommendation_agent import TeachingRecommendationAgent

def test_agent_generates_explainable_suggestions():
    agent = TeachingRecommendationAgent(use_local=True)
    
    analytics = {
        "session_id": "TEST_SES_01",
        "attendance_percentage": 75.0,
        "average_engagement": 42.0,
        "audio_activity_percentage": 18.0,  # Low audio
        "attention_drops": [
            {"time_range": "05:00 - 06:00", "drop_points": 24.0, "engagement_score": 38.0}
        ],
        "high_activity_periods": [],
        "long_inactivity_periods": [
            {"time_range": "08:00 - 09:00", "engagement_score": 35.0}
        ],
        "transient_count": 2,
        "session_duration_sec": 600.0
    }
    
    recs = agent.generate_recommendations("TEST_SES_01", analytics)
    
    assert len(recs) >= 3
    for r in recs:
        assert "title" in r
        assert "observation" in r
        assert "reasoning" in r  # Pedagogical WHY
        assert "suggestion" in r  # Actionable advice
        assert len(r["reasoning"]) > 15
        assert len(r["suggestion"]) > 15

    # Check specific rule triggers
    titles = [r["title"] for r in recs]
    assert any("Attention Drops" in t for t in titles)
    assert any("Two-Way Dialogue" in t for t in titles)
    assert any("Transient Visitor" in t for t in titles)
