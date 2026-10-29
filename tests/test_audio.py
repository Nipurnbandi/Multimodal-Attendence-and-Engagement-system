"""
Unit tests for Audio Processing, RMS Energy, and VAD Segmentation.
"""

import os
import tempfile
import numpy as np
import scipy.io.wavfile as wavfile
import pytest
from app.audio.processor import ClassroomAudioProcessor

def test_audio_processor_empty_fallback():
    processor = ClassroomAudioProcessor()
    profile = processor.analyze_audio_file(None, fallback_duration_sec=30.0)
    assert profile["has_audio"] is False
    assert profile["duration_sec"] == 30.0
    assert profile["total_speech_sec"] == 0.0
    assert profile["audio_activity_percentage"] == 0.0

def test_audio_processor_synthetic_analysis():
    processor = ClassroomAudioProcessor(target_sample_rate=16000, window_sec=0.5)
    
    # Create temporary WAV with 2 seconds silence + 2 seconds tone + 1 second silence
    sr = 16000
    t = np.arange(5 * sr) / sr
    audio = np.zeros(5 * sr, dtype=np.int16)
    
    # 2.0 to 4.0 is active speech tone (~200Hz)
    active_mask = (t >= 2.0) & (t <= 4.0)
    tone = (0.5 * np.sin(2 * np.pi * 200 * t) * 32767).astype(np.int16)
    audio[active_mask] = tone[active_mask]
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        wavfile.write(tmp_path, sr, audio)
        
    try:
        profile = processor.analyze_audio_file(tmp_path)
        assert profile["has_audio"] is True
        assert abs(profile["duration_sec"] - 5.0) < 0.2
        assert profile["total_speech_sec"] > 1.0
        assert profile["audio_activity_percentage"] > 20.0
        assert profile["activity_segments_count"] >= 1
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
