"""
Configuration settings for the Multimodal Attendance & Engagement Agent.
Manages application paths, thresholds, algorithmic weights, and privacy flags.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"
SAMPLES_DIR = DATA_DIR / "samples"

# Ensure directories exist
for folder in [INPUT_DIR, PROCESSED_DIR, REPORTS_DIR, SAMPLES_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

@dataclass
class SystemConfig:
    # Database
    db_path: Path = DATA_DIR / "attendance_engagement.db"
    database_url: str = field(default_factory=lambda: f"sqlite:///{DATA_DIR / 'attendance_engagement.db'}")
    
    # Vision & Attendance Parameters
    sustained_presence_threshold_sec: float = 10.0  # Min duration in seconds to consider presence sustained
    presence_ratio_threshold: float = 0.35          # Min ratio of session length active to qualify as Present
    min_detection_confidence: float = 0.50          # Face detection confidence
    frame_sample_rate_fps: float = 2.0             # Sample frames at 2 FPS for fast, accurate laptop processing
    tracker_max_disappeared_frames: int = 15       # Max missed frames before terminating a track
    tracker_distance_threshold: float = 80.0       # Pixel distance threshold for centroid tracking
    
    # Engagement Calculation Weights (Must sum to 1.0)
    # Engagement Score = w1*Gaze + w2*Movement + w3*Audio + w4*Continuity
    weight_frontal_attention: float = 0.40
    weight_movement_activity: float = 0.25
    weight_audio_activity: float = 0.20
    weight_presence_continuity: float = 0.15
    
    # Session Analytics Parameters
    interval_duration_sec: float = 30.0            # Dynamics calculated per 30-second window
    attention_drop_threshold: float = 20.0         # Drop of >20 points indicates potential attention lapse
    low_engagement_threshold: float = 40.0         # Absolute score < 40 considered low engagement indicator
    high_engagement_threshold: float = 75.0        # Absolute score > 75 considered high engagement indicator
    
    # Audio Analysis Parameters
    audio_sample_rate: int = 16000
    audio_energy_threshold_multiplier: float = 1.3 # Factor above noise floor for speech activity
    audio_window_duration_sec: float = 0.5         # Window for audio energy analysis
    
    # Privacy & Ethics Controls
    anonymize_participants: bool = True            # Anonymize student identities by default
    store_raw_video: bool = False                  # Do not retain full video copies permanently
    retention_period_days: int = 30                # Session analytics retention
    
    # LLM & Agent Settings
    use_local_reasoning: bool = True               # Offline expert system default
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

# Singleton instance
config = SystemConfig()
