"""
Multimodal Engagement Estimation and Session Dynamics Engine.
Combines visual behavioral signals (head pose orientation, eye gaze alertness, micro-movement)
with acoustic activity (speech interaction, silence ratio) into transparent, interval-based indicators.

DISCLAIMER:
All engagement metrics produced by this module represent estimated behavioral indicators.
They provide assistive decision support for educators and do not claim to measure cognitive attention directly.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from app.config import config
from app.vision.tracker import ParticipantTrack

class MultimodalEngagementEngine:
    """
    Computes explainable multimodal engagement scores and divides sessions into dynamic time intervals.
    """

    def __init__(
        self,
        w_attention: float = config.weight_frontal_attention,
        w_motion: float = config.weight_movement_activity,
        w_audio: float = config.weight_audio_activity,
        w_presence: float = config.weight_presence_continuity,
        interval_sec: float = config.interval_duration_sec
    ):
        # Normalize weights to ensure sum = 1.0
        total_w = w_attention + w_motion + w_audio + w_presence
        self.w_attention = w_attention / total_w
        self.w_motion = w_motion / total_w
        self.w_audio = w_audio / total_w
        self.w_presence = w_presence / total_w
        self.interval_sec = interval_sec

    def compute_instantaneous_engagement(
        self,
        attention_signal: float,   # 0 - 100
        movement_score: float,     # 0 - 100
        audio_active: bool,        # True / False
        presence_continuity: float # 0 - 100
    ) -> float:
        """
        Calculates instantaneous engagement indicator:
        E = w1 * Attention + w2 * NormalizedMovement + w3 * AudioSignal + w4 * Continuity
        """
        # Optimal alert movement is moderate (around 20-50).
        # We penalize both complete immobility (<5) and frantic restlessness (>85).
        if movement_score < 5.0:
            norm_motion = 40.0  # Stillness penalty
        elif movement_score > 85.0:
            norm_motion = 50.0  # Restlessness penalty
        else:
            # Active alertness
            norm_motion = min(100.0, movement_score * 1.5)

        audio_signal = 85.0 if audio_active else 45.0  # Background attentiveness during lecture

        score = (
            self.w_attention * attention_signal +
            self.w_motion * norm_motion +
            self.w_audio * audio_signal +
            self.w_presence * presence_continuity
        )
        return float(np.clip(score, 0.0, 100.0))

    def analyze_session_intervals(
        self,
        session_duration_sec: float,
        tracks: Dict[str, ParticipantTrack],
        timeline_visual: List[Dict[str, Any]],
        audio_profile: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Partitions the session into discrete intervals (e.g. 30s or 60s)
        and computes aggregate dynamics, attention drops, and interaction peaks.
        """
        if session_duration_sec <= 0:
            session_duration_sec = 60.0

        num_intervals = max(1, int(np.ceil(session_duration_sec / self.interval_sec)))
        interval_metrics_list: List[Dict[str, Any]] = []

        audio_timeline = audio_profile.get("timeline_audio_records", [])

        prev_engagement = None
        attention_drops = []
        interaction_peaks = []
        long_inactivity_periods = []

        total_enrolled_estimate = max(1, len(tracks))

        for idx in range(num_intervals):
            start_sec = idx * self.interval_sec
            end_sec = min(session_duration_sec, (idx + 1) * self.interval_sec)
            duration_of_interval = max(0.1, end_sec - start_sec)

            # Filter visual records in this interval
            v_recs = [r for r in timeline_visual if start_sec <= r["timestamp_sec"] < end_sec]

            # Filter audio records in this interval
            a_recs = [r for r in audio_timeline if start_sec <= r["timestamp_sec"] < end_sec]

            # 1. Active Participants in interval
            active_ids = set()
            for rec in v_recs:
                active_ids.update(rec.get("tracked_ids", []))
            active_count = len(active_ids)
            inactive_count = max(0, total_enrolled_estimate - active_count)

            # 2. Average Visual Signals
            if v_recs:
                avg_att = float(np.mean([r["avg_attention_signal"] for r in v_recs]))
                avg_mot = float(np.mean([r["avg_movement_score"] for r in v_recs]))
            else:
                avg_att = 50.0
                avg_mot = 25.0

            # 3. Audio Activity in interval
            if a_recs:
                active_audio_count = sum(1 for r in a_recs if r.get("is_active", False))
                audio_ratio = active_audio_count / len(a_recs)
            else:
                audio_ratio = 0.0

            # 4. Presence Continuity Ratio
            presence_ratio = (active_count / total_enrolled_estimate) * 100.0

            # 5. Combined Engagement Score
            interval_engagement = (
                self.w_attention * avg_att +
                self.w_motion * avg_mot +
                self.w_audio * (audio_ratio * 100.0) +
                self.w_presence * presence_ratio
            )
            interval_engagement = float(np.clip(interval_engagement, 0.0, 100.0))

            # 6. Flag Detection
            attention_drop = False
            if prev_engagement is not None:
                drop_amount = prev_engagement - interval_engagement
                if drop_amount >= config.attention_drop_threshold:
                    attention_drop = True
                    attention_drops.append({
                        "interval_index": idx,
                        "time_range": f"{int(start_sec//60):02d}:{int(start_sec%60):02d} - {int(end_sec//60):02d}:{int(end_sec%60):02d}",
                        "drop_points": round(drop_amount, 1),
                        "engagement_score": round(interval_engagement, 1)
                    })

            interaction_peak = False
            if audio_ratio >= 0.50 or interval_engagement >= config.high_engagement_threshold:
                interaction_peak = True
                interaction_peaks.append({
                    "interval_index": idx,
                    "time_range": f"{int(start_sec//60):02d}:{int(start_sec%60):02d} - {int(end_sec//60):02d}:{int(end_sec%60):02d}",
                    "engagement_score": round(interval_engagement, 1),
                    "audio_activity_pct": round(audio_ratio * 100.0, 1)
                })

            if interval_engagement < config.low_engagement_threshold and audio_ratio < 0.15:
                long_inactivity_periods.append({
                    "interval_index": idx,
                    "time_range": f"{int(start_sec//60):02d}:{int(start_sec%60):02d} - {int(end_sec//60):02d}:{int(end_sec%60):02d}",
                    "engagement_score": round(interval_engagement, 1)
                })

            metric_entry = {
                "interval_index": idx,
                "start_sec": round(start_sec, 1),
                "end_sec": round(end_sec, 1),
                "timestamp_label": f"{int(start_sec//60):02d}:{int(start_sec%60):02d} - {int(end_sec//60):02d}:{int(end_sec%60):02d}",
                "active_participants": int(active_count),
                "inactive_participants": int(inactive_count),
                "avg_engagement_score": round(interval_engagement, 2),
                "avg_attention_signal": round(avg_att, 2),
                "avg_movement_activity": round(avg_mot, 2),
                "audio_activity_ratio": round(audio_ratio, 3),
                "attention_drop_flag": attention_drop,
                "interaction_peak_flag": interaction_peak
            }
            interval_metrics_list.append(metric_entry)
            prev_engagement = interval_engagement

        # Overall Session Dynamics
        all_engagements = [m["avg_engagement_score"] for m in interval_metrics_list]
        overall_avg_engagement = float(np.mean(all_engagements)) if all_engagements else 0.0

        dynamics_summary = {
            "overall_avg_engagement": round(overall_avg_engagement, 2),
            "peak_engagement": round(float(np.max(all_engagements)), 2) if all_engagements else 0.0,
            "lowest_engagement": round(float(np.min(all_engagements)), 2) if all_engagements else 0.0,
            "attention_drops": attention_drops,
            "interaction_peaks": interaction_peaks,
            "long_inactivity_periods": long_inactivity_periods,
            "total_intervals": num_intervals,
            "formula_documentation": (
                f"Engagement = {self.w_attention:.2f}*FrontalAttention + "
                f"{self.w_motion:.2f}*MovementActivity + "
                f"{self.w_audio:.2f}*AudioActivity + "
                f"{self.w_presence:.2f}*PresenceContinuity"
            )
        }

        return interval_metrics_list, dynamics_summary
