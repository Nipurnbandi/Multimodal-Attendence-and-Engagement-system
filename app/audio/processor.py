"""
Classroom Audio Signal Processing Module.
Offline-capable extraction, Root-Mean-Square (RMS) energy calculation,
Zero-Crossing Rate (ZCR), Voice Activity Detection (VAD), and silence segmentation.
Preserves privacy by analyzing signal features only, without biometric identification or transcription.
"""

import os
import subprocess
import tempfile
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import scipy.io.wavfile as wavfile
from app.config import config, PROCESSED_DIR

try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_EXE = "ffmpeg"


class ClassroomAudioProcessor:
    """
    Offline signal processing pipeline for classroom audio.
    Extracts speech activity, interaction segments, and silence ratios.
    """

    def __init__(self, target_sample_rate: int = 16000, window_sec: float = 0.5):
        self.sample_rate = target_sample_rate
        self.window_sec = window_sec
        self.window_size = int(self.sample_rate * self.window_sec)

    def extract_audio_from_video(self, video_path: str, output_wav_path: Optional[str] = None) -> Optional[str]:
        """
        Extracts 16kHz mono WAV audio from an input video container using bundled FFmpeg.
        Returns path to WAV file, or None if video has no audio track.
        """
        if output_wav_path is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_wav_path = str(PROCESSED_DIR / f"{base_name}_audio.wav")

        cmd = [
            FFMPEG_EXE,
            "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(self.sample_rate),
            "-ac", "1",
            output_wav_path
        ]

        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
            if result.returncode == 0 and os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 100:
                return output_wav_path
            else:
                return None
        except Exception:
            return None

    def analyze_audio_file(self, wav_path: Optional[str], fallback_duration_sec: float = 0.0) -> Dict[str, Any]:
        """
        Performs signal-level feature extraction on a WAV audio file.
        Returns comprehensive speech/silence metrics and temporal activity segments.
        """
        if not wav_path or not os.path.exists(wav_path):
            # Fallback for silent video or videos without audio tracks
            return self._generate_empty_audio_profile(fallback_duration_sec)

        try:
            sr, signal = wavfile.read(wav_path)
            # Convert to float32 normalized [-1.0, 1.0]
            if signal.dtype == np.int16:
                signal = signal.astype(np.float32) / 32768.0
            elif signal.dtype == np.int32:
                signal = signal.astype(np.float32) / 2147483648.0
            elif signal.dtype == np.uint8:
                signal = (signal.astype(np.float32) - 128.0) / 128.0

            total_samples = len(signal)
            duration_sec = total_samples / sr if sr > 0 else fallback_duration_sec

            if duration_sec <= 0:
                return self._generate_empty_audio_profile(fallback_duration_sec)

            # Slide window across signal
            num_windows = total_samples // self.window_size
            if num_windows == 0:
                return self._generate_empty_audio_profile(duration_sec)

            rms_values = []
            zcr_values = []
            timestamps = []

            for i in range(num_windows):
                start_idx = i * self.window_size
                end_idx = start_idx + self.window_size
                chunk = signal[start_idx:end_idx]

                # RMS Energy
                rms = np.sqrt(np.mean(chunk**2))
                rms_values.append(rms)

                # Zero Crossing Rate
                zero_crossings = np.sum(np.abs(np.diff(np.sign(chunk)))) / (2.0 * len(chunk))
                zcr_values.append(zero_crossings)

                timestamps.append(i * self.window_sec)

            rms_arr = np.array(rms_values)
            zcr_arr = np.array(zcr_values)

            # Adaptive energy threshold:
            # Baseline ambient noise floor calculated from 20th percentile
            noise_floor = np.percentile(rms_arr, 20)
            energy_threshold = max(0.005, noise_floor * config.audio_energy_threshold_multiplier)

            # Activity decision: RMS > threshold AND non-trivial ZCR
            active_mask = (rms_arr > energy_threshold) & (zcr_arr > 0.015) & (zcr_arr < 0.45)

            # Compute timeline records
            timeline_audio_records = []
            for t, is_act, energy, zcr in zip(timestamps, active_mask, rms_arr, zcr_arr):
                timeline_audio_records.append({
                    "timestamp_sec": round(float(t), 2),
                    "is_active": bool(is_act),
                    "energy_rms": round(float(energy), 5),
                    "zcr": round(float(zcr), 4)
                })

            # Detect continuous activity segments
            segments = self._extract_continuous_segments(active_mask, self.window_sec)

            active_windows_count = int(np.sum(active_mask))
            total_speech_sec = active_windows_count * self.window_sec
            total_silence_sec = max(0.0, duration_sec - total_speech_sec)
            activity_percentage = (total_speech_sec / duration_sec) * 100.0 if duration_sec > 0 else 0.0

            return {
                "has_audio": True,
                "duration_sec": round(duration_sec, 2),
                "total_speech_sec": round(total_speech_sec, 2),
                "total_silence_sec": round(total_silence_sec, 2),
                "audio_activity_percentage": round(activity_percentage, 1),
                "activity_segments_count": len(segments),
                "activity_segments": segments,
                "timeline_audio_records": timeline_audio_records,
                "noise_floor": round(float(noise_floor), 5),
                "energy_threshold": round(float(energy_threshold), 5)
            }

        except Exception as e:
            return self._generate_empty_audio_profile(fallback_duration_sec, error=str(e))

    def _extract_continuous_segments(self, active_mask: np.ndarray, step_sec: float) -> List[Dict[str, float]]:
        """Identifies continuous intervals of classroom audio activity (questions, discussions, lecture explanations)."""
        segments = []
        in_segment = False
        seg_start = 0.0

        for idx, is_act in enumerate(active_mask):
            current_time = idx * step_sec
            if is_act and not in_segment:
                in_segment = True
                seg_start = current_time
            elif not is_act and in_segment:
                in_segment = False
                seg_duration = current_time - seg_start
                if seg_duration >= 1.0:  # Ignore clicks/pops shorter than 1s
                    segments.append({
                        "start_sec": round(seg_start, 2),
                        "end_sec": round(current_time, 2),
                        "duration_sec": round(seg_duration, 2)
                    })

        if in_segment:
            segments.append({
                "start_sec": round(seg_start, 2),
                "end_sec": round(len(active_mask) * step_sec, 2),
                "duration_sec": round(len(active_mask) * step_sec - seg_start, 2)
            })

        return segments

    def _generate_empty_audio_profile(self, duration_sec: float, error: Optional[str] = None) -> Dict[str, Any]:
        """Generates fallback profile when audio track is absent or empty."""
        return {
            "has_audio": False,
            "duration_sec": duration_sec,
            "total_speech_sec": 0.0,
            "total_silence_sec": duration_sec,
            "audio_activity_percentage": 0.0,
            "activity_segments_count": 0,
            "activity_segments": [],
            "timeline_audio_records": [],
            "noise_floor": 0.0,
            "energy_threshold": 0.0,
            "notes": error or "No audio stream detected in media file; using visual-only behavioral estimation."
        }
