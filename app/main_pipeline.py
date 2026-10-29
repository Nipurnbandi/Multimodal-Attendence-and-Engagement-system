"""
Master Pipeline Coordinator.
Orchestrates the entire Multimodal Attendance & Engagement Agent workflow:
Video Ingestion -> Audio Processing -> Computer Vision Tracking -> Multimodal Fusion
-> Attendance Logic -> Interval Dynamics -> AI Reasoning -> Database Persistence.
"""

import os
import uuid
from datetime import datetime
from typing import Dict, Any, Callable, Optional
from app.config import config
from app.vision.video_processor import VideoProcessor
from app.audio.processor import ClassroomAudioProcessor
from app.attendance.manager import AttendanceManager
from app.engagement.engine import MultimodalEngagementEngine
from app.agent.recommendation_agent import TeachingRecommendationAgent
from app.analytics.reporter import SessionAnalyticsReporter
from app.database.models import Session, AttendanceRecord, IntervalMetric, SessionRecommendation
from app.database.db import get_db

class MultimodalPipelineCoordinator:
    """
    High-level orchestrator for processing lectures and classroom sessions.
    """

    def __init__(self):
        self.video_processor = VideoProcessor(sample_fps=config.frame_sample_rate_fps)
        self.audio_processor = ClassroomAudioProcessor()
        self.attendance_manager = AttendanceManager(sustained_threshold_sec=config.sustained_presence_threshold_sec)
        self.engagement_engine = MultimodalEngagementEngine()
        self.agent = TeachingRecommendationAgent(use_local=config.use_local_reasoning)

    def run_pipeline(
        self,
        video_path: str,
        session_title: str = "Classroom Lecture",
        course_code: str = "CS-401",
        instructor_name: str = "Faculty",
        source_type: str = "uploaded_video",
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Executes complete multimodal analysis on a classroom video.
        """
        session_id = f"SES-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

        if progress_callback:
            progress_callback(0.05, "Step 1/5: Extracting and analyzing audio signals...")

        # 1. Audio Analysis
        extracted_wav = self.audio_processor.extract_audio_from_video(video_path)
        audio_profile = self.audio_processor.analyze_audio_file(extracted_wav)

        if progress_callback:
            progress_callback(0.20, "Step 2/5: Processing video frames, tracking participants & head pose...")

        # 2. Vision Processing
        def vision_subprogress(pct, msg):
            if progress_callback:
                # Map vision progress to 0.20 - 0.65
                overall_pct = 0.20 + (pct * 0.45)
                progress_callback(overall_pct, f"Step 2/5: {msg}")

        vision_results = self.video_processor.process_video_file(video_path, progress_callback=vision_subprogress)

        duration_sec = vision_results["duration_sec"]
        if duration_sec <= 0:
            duration_sec = audio_profile.get("duration_sec", 60.0)

        if progress_callback:
            progress_callback(0.70, "Step 3/5: Evaluating sustained presence and finalizing attendance...")

        # 3. Attendance Evaluation
        attendance_records, attendance_summary = self.attendance_manager.evaluate_attendance(
            session_id=session_id,
            tracks=vision_results["tracks"],
            session_duration_sec=duration_sec
        )

        if progress_callback:
            progress_callback(0.80, "Step 4/5: Fusing multimodal signals and calculating interval dynamics...")

        # 4. Multimodal Engagement & Intervals
        interval_metrics, dynamics_summary = self.engagement_engine.analyze_session_intervals(
            session_duration_sec=duration_sec,
            tracks=vision_results["tracks"],
            timeline_visual=vision_results["timeline_visual_records"],
            audio_profile=audio_profile
        )

        if progress_callback:
            progress_callback(0.90, "Step 5/5: Generating AI pedagogical recommendations and saving results...")

        # 5. AI Teaching Recommendations
        structured_analytics = {
            "session_id": session_id,
            "session_duration_sec": duration_sec,
            "attendance_percentage": attendance_summary["attendance_percentage"],
            "average_engagement": dynamics_summary["overall_avg_engagement"],
            "low_engagement_periods": dynamics_summary["long_inactivity_periods"],
            "high_activity_periods": dynamics_summary["interaction_peaks"],
            "audio_activity_percentage": audio_profile["audio_activity_percentage"],
            "long_inactivity_periods": dynamics_summary["long_inactivity_periods"],
            "attention_drops": dynamics_summary["attention_drops"],
            "transient_count": attendance_summary["transient_count"]
        }

        recommendations = self.agent.generate_recommendations(session_id, structured_analytics)

        # 6. Database Persistence
        session_meta = {
            "id": session_id,
            "title": session_title,
            "course_code": course_code,
            "instructor_name": instructor_name,
            "source_type": source_type,
            "video_filename": os.path.basename(video_path),
            "duration_seconds": duration_sec,
            "total_frames": vision_results["total_frames"],
            "processed_frames": vision_results["processed_frames"],
            "status": "completed",
            "average_engagement": dynamics_summary["overall_avg_engagement"],
            "attendance_percentage": attendance_summary["attendance_percentage"],
            "created_at": datetime.utcnow()
        }

        self._persist_session_to_db(
            session_meta=session_meta,
            attendance_records=attendance_records,
            interval_metrics=interval_metrics,
            recommendations=recommendations
        )

        # 7. Export Reports
        csv_path = SessionAnalyticsReporter.export_attendance_csv(session_id, attendance_records)
        json_payload = {
            "session_meta": session_meta,
            "attendance_summary": attendance_summary,
            "attendance_records": attendance_records,
            "audio_profile": {
                "has_audio": audio_profile.get("has_audio", False),
                "total_speech_sec": audio_profile.get("total_speech_sec", 0.0),
                "total_silence_sec": audio_profile.get("total_silence_sec", 0.0),
                "audio_activity_percentage": audio_profile.get("audio_activity_percentage", 0.0),
                "activity_segments_count": audio_profile.get("activity_segments_count", 0)
            },
            "dynamics_summary": dynamics_summary,
            "interval_metrics": interval_metrics,
            "recommendations": recommendations
        }
        json_path = SessionAnalyticsReporter.export_full_analytics_json(json_payload)

        if progress_callback:
            progress_callback(1.0, "Session analysis completed successfully!")

        return {
            "session_id": session_id,
            "session_meta": session_meta,
            "attendance_summary": attendance_summary,
            "attendance_records": attendance_records,
            "audio_profile": audio_profile,
            "dynamics_summary": dynamics_summary,
            "interval_metrics": interval_metrics,
            "recommendations": recommendations,
            "preview_frames": vision_results.get("preview_frames", []),
            "csv_path": csv_path,
            "json_path": json_path
        }

    def _persist_session_to_db(
        self,
        session_meta: Dict[str, Any],
        attendance_records: List[Dict[str, Any]],
        interval_metrics: List[Dict[str, Any]],
        recommendations: List[Dict[str, Any]]
    ):
        """Saves all session entities into SQLite database."""
        with get_db() as db:
            session_obj = Session(
                id=session_meta["id"],
                title=session_meta["title"],
                course_code=session_meta["course_code"],
                instructor_name=session_meta["instructor_name"],
                source_type=session_meta["source_type"],
                video_filename=session_meta["video_filename"],
                duration_seconds=session_meta["duration_seconds"],
                total_frames=session_meta["total_frames"],
                processed_frames=session_meta["processed_frames"],
                status="completed",
                average_engagement=session_meta["average_engagement"],
                attendance_percentage=session_meta["attendance_percentage"],
                created_at=session_meta["created_at"]
            )
            db.add(session_obj)
            db.flush()

            for rec in attendance_records:
                db.add(AttendanceRecord(
                    session_id=session_meta["id"],
                    participant_id=rec["participant_id"],
                    participant_name=rec.get("participant_name", "Participant"),
                    first_seen_sec=rec["first_seen_sec"],
                    last_seen_sec=rec["last_seen_sec"],
                    total_detected_sec=rec["total_detected_sec"],
                    presence_ratio=rec["presence_ratio"],
                    attendance_status=rec["attendance_status"],
                    confidence_score=rec["confidence_score"],
                    notes=rec.get("notes", "")
                ))

            for metric in interval_metrics:
                db.add(IntervalMetric(
                    session_id=session_meta["id"],
                    interval_index=metric["interval_index"],
                    start_sec=metric["start_sec"],
                    end_sec=metric["end_sec"],
                    active_participants=metric["active_participants"],
                    inactive_participants=metric["inactive_participants"],
                    avg_engagement_score=metric["avg_engagement_score"],
                    avg_attention_signal=metric["avg_attention_signal"],
                    avg_movement_activity=metric["avg_movement_activity"],
                    audio_activity_ratio=metric["audio_activity_ratio"],
                    attention_drop_flag=metric["attention_drop_flag"],
                    interaction_peak_flag=metric["interaction_peak_flag"]
                ))

            for r in recommendations:
                db.add(SessionRecommendation(
                    session_id=session_meta["id"],
                    category=r["category"],
                    timestamp_range=r.get("timestamp_range", "Session"),
                    severity=r.get("severity", "info"),
                    title=r["title"],
                    observation=r["observation"],
                    reasoning=r["reasoning"],
                    suggestion=r["suggestion"]
                ))
            db.commit()
