"""
SQLAlchemy Database Models for Multimodal Attendance & Engagement Agent.
Defines schemas for Students, Sessions, Attendance Records, Interval Metrics, and AI Recommendations.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

def utc_now():
    return datetime.now(timezone.utc)

class Student(Base):
    """Registered or enrolled student record."""
    __tablename__ = "students"

    id = Column(String(64), primary_key=True)
    name = Column(String(128), nullable=False)
    roll_number = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=utc_now)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "roll_number": self.roll_number,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class Session(Base):
    """Lecture or training classroom session."""
    __tablename__ = "sessions"

    id = Column(String(64), primary_key=True)
    title = Column(String(256), nullable=False)
    course_code = Column(String(64), nullable=True, default="UNIV-101")
    instructor_name = Column(String(128), nullable=True, default="Instructor")
    source_type = Column(String(64), default="uploaded_video")  # uploaded_video, webcam_live, sample_demo
    video_filename = Column(String(256), nullable=True)
    duration_seconds = Column(Float, default=0.0)
    total_frames = Column(Integer, default=0)
    processed_frames = Column(Integer, default=0)
    status = Column(String(32), default="pending")  # pending, processing, completed, failed
    average_engagement = Column(Float, default=0.0)
    attendance_percentage = Column(Float, default=0.0)
    created_at = Column(DateTime, default=utc_now)

    # Relationships
    attendance_records = relationship("AttendanceRecord", back_populates="session", cascade="all, delete-orphan")
    interval_metrics = relationship("IntervalMetric", back_populates="session", cascade="all, delete-orphan")
    recommendations = relationship("SessionRecommendation", back_populates="session", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "course_code": self.course_code,
            "instructor_name": self.instructor_name,
            "source_type": self.source_type,
            "video_filename": self.video_filename,
            "duration_seconds": self.duration_seconds,
            "status": self.status,
            "average_engagement": round(self.average_engagement, 2) if self.average_engagement else 0.0,
            "attendance_percentage": round(self.attendance_percentage, 1) if self.attendance_percentage else 0.0,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class AttendanceRecord(Base):
    """Individual participant attendance calculation based on sustained presence."""
    __tablename__ = "attendance_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("sessions.id"), nullable=False)
    participant_id = Column(String(64), nullable=False)
    participant_name = Column(String(128), default="Participant")
    first_seen_sec = Column(Float, default=0.0)
    last_seen_sec = Column(Float, default=0.0)
    total_detected_sec = Column(Float, default=0.0)
    presence_ratio = Column(Float, default=0.0)
    attendance_status = Column(String(32), default="Absent")  # Present, Absent, Transient
    confidence_score = Column(Float, default=0.0)
    notes = Column(Text, nullable=True)

    session = relationship("Session", back_populates="attendance_records")

    def to_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "participant_id": self.participant_id,
            "participant_name": self.participant_name,
            "first_seen_sec": round(self.first_seen_sec, 2),
            "last_seen_sec": round(self.last_seen_sec, 2),
            "total_detected_sec": round(self.total_detected_sec, 2),
            "presence_ratio": round(self.presence_ratio, 3),
            "attendance_status": self.attendance_status,
            "confidence_score": round(self.confidence_score, 2),
            "notes": self.notes
        }

class IntervalMetric(Base):
    """Dynamics and multimodal indicators aggregated across discrete time slices (e.g. 30s or 60s)."""
    __tablename__ = "interval_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("sessions.id"), nullable=False)
    interval_index = Column(Integer, nullable=False)
    start_sec = Column(Float, nullable=False)
    end_sec = Column(Float, nullable=False)
    active_participants = Column(Integer, default=0)
    inactive_participants = Column(Integer, default=0)
    avg_engagement_score = Column(Float, default=0.0)
    avg_attention_signal = Column(Float, default=0.0)
    avg_movement_activity = Column(Float, default=0.0)
    audio_activity_ratio = Column(Float, default=0.0)
    attention_drop_flag = Column(Boolean, default=False)
    interaction_peak_flag = Column(Boolean, default=False)

    session = relationship("Session", back_populates="interval_metrics")

    def to_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "interval_index": self.interval_index,
            "start_sec": round(self.start_sec, 1),
            "end_sec": round(self.end_sec, 1),
            "timestamp_label": f"{int(self.start_sec // 60):02d}:{int(self.start_sec % 60):02d} - {int(self.end_sec // 60):02d}:{int(self.end_sec % 60):02d}",
            "active_participants": self.active_participants,
            "inactive_participants": self.inactive_participants,
            "avg_engagement_score": round(self.avg_engagement_score, 2),
            "avg_attention_signal": round(self.avg_attention_signal, 2),
            "avg_movement_activity": round(self.avg_movement_activity, 2),
            "audio_activity_ratio": round(self.audio_activity_ratio, 2),
            "attention_drop_flag": self.attention_drop_flag,
            "interaction_peak_flag": self.interaction_peak_flag
        }

class SessionRecommendation(Base):
    """AI-generated pedagogical teaching suggestions and actionable feedback."""
    __tablename__ = "session_recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("sessions.id"), nullable=False)
    category = Column(String(64), default="Teaching Suggestion")
    timestamp_range = Column(String(64), nullable=True)
    severity = Column(String(32), default="info")  # info, warning, action_suggested
    title = Column(String(256), nullable=False)
    observation = Column(Text, nullable=False)
    reasoning = Column(Text, nullable=False)
    suggestion = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utc_now)

    session = relationship("Session", back_populates="recommendations")

    def to_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "category": self.category,
            "timestamp_range": self.timestamp_range,
            "severity": self.severity,
            "title": self.title,
            "observation": self.observation,
            "reasoning": self.reasoning,
            "suggestion": self.suggestion,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
