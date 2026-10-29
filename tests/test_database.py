"""
Unit tests for Database Operations, Schema Models, and Privacy Deletion.
"""

import pytest
from app.database.db import get_db, delete_session_data, init_db, seed_sample_students
from app.database.models import Student, Session, AttendanceRecord

def test_database_persistence_and_deletion():
    init_db()
    seed_sample_students()
    
    test_id = "SES_TEST_DB_001"
    
    with get_db() as db:
        session_obj = Session(
            id=test_id,
            title="Database Unit Test Session",
            course_code="TEST101",
            instructor_name="Prof. Tester",
            source_type="unit_test",
            duration_seconds=120.0,
            status="completed",
            average_engagement=68.5,
            attendance_percentage=85.0
        )
        db.add(session_obj)
        
        att_rec = AttendanceRecord(
            session_id=test_id,
            participant_id="STU-001",
            participant_name="Alex Rivera",
            attendance_status="Present",
            total_detected_sec=100.0,
            confidence_score=0.92
        )
        db.add(att_rec)
        db.commit()

    # Query back
    with get_db() as db:
        s = db.query(Session).filter_by(id=test_id).first()
        assert s is not None
        assert s.title == "Database Unit Test Session"
        assert len(s.attendance_records) == 1
        assert s.attendance_records[0].attendance_status == "Present"

    # Test Privacy Deletion
    deleted = delete_session_data(test_id)
    assert deleted is True

    # Verify records purged
    with get_db() as db:
        s = db.query(Session).filter_by(id=test_id).first()
        assert s is None
        att_count = db.query(AttendanceRecord).filter_by(session_id=test_id).count()
        assert att_count == 0
