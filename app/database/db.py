"""
Database engine and session management.
Provides thread-safe session factories, schema initialization, and data access helpers.
"""

from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session as SASession
from app.config import config
from app.database.models import Base, Student, Session, AttendanceRecord, IntervalMetric, SessionRecommendation

# SQLite with check_same_thread=False for Streamlit multi-threading compatibility
engine = create_engine(
    config.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in config.database_url else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Create all tables in the database if they do not exist."""
    Base.metadata.create_all(bind=engine)

@contextmanager
def get_db():
    """Provide a transactional database session scope."""
    db: SASession = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def seed_sample_students():
    """Ensure baseline registered students exist for matching and demonstration."""
    with get_db() as db:
        existing = db.query(Student).count()
        if existing == 0:
            sample_roster = [
                Student(id="STU-001", name="Alex Rivera", roll_number="CS-2026-01"),
                Student(id="STU-002", name="Priya Sharma", roll_number="CS-2026-02"),
                Student(id="STU-003", name="Liam Chen", roll_number="CS-2026-03"),
                Student(id="STU-004", name="Fatima Al-Mansoor", roll_number="CS-2026-04"),
                Student(id="STU-005", name="Marcus Johnson", roll_number="CS-2026-05"),
                Student(id="STU-006", name="Elena Rostova", roll_number="CS-2026-06"),
                Student(id="STU-007", name="Noah Williams", roll_number="CS-2026-07"),
                Student(id="STU-008", name="Aoi Takahashi", roll_number="CS-2026-08"),
            ]
            db.add_all(sample_roster)
            db.commit()

def delete_session_data(session_id: str) -> bool:
    """Privacy requirement: delete raw and derived session records upon instructor request."""
    with get_db() as db:
        session_obj = db.query(Session).filter(Session.id == session_id).first()
        if session_obj:
            db.delete(session_obj)
            db.commit()
            return True
        return False
