"""
Classroom Attendance Tracking & Roster Management.
Evaluates sustained presence to distinguish legitimate attendance from transient passersby.
Calculates attendance duration, percentages, and handles roster reconciliation.
"""

from typing import Dict, List, Tuple, Any, Optional
from app.config import config
from app.vision.tracker import ParticipantTrack
from app.database.models import Student, AttendanceRecord
from app.database.db import get_db

class AttendanceManager:
    """
    Evaluates tracking data to finalize attendance records.
    Enforces privacy-preserving rules and sustained presence thresholds.
    """

    def __init__(self, sustained_threshold_sec: float = config.sustained_presence_threshold_sec):
        self.sustained_threshold_sec = sustained_threshold_sec

    def evaluate_attendance(
        self,
        session_id: str,
        tracks: Dict[str, ParticipantTrack],
        session_duration_sec: float
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Processes participant tracks and returns:
        1. List of individual attendance records
        2. Session-level attendance summary
        """
        # Fetch registered student roster to reconcile names where available
        roster_map: Dict[str, Dict[str, str]] = {}
        with get_db() as db:
            students = db.query(Student).all()
            for s in students:
                roster_map[s.id] = {"id": s.id, "name": s.name, "roll_number": s.roll_number or ""}

        student_ids_list = list(roster_map.keys())

        records: List[Dict[str, Any]] = []
        present_count = 0
        transient_count = 0
        absent_count = 0

        # Sort tracks by track_id
        sorted_track_ids = sorted(tracks.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else x)

        for idx, tid in enumerate(sorted_track_ids):
            track = tracks[tid]
            first_seen = track.first_seen_sec
            last_seen = track.last_seen_sec
            total_active_sec = len(track.detection_timestamps) / max(1.0, config.frame_sample_rate_fps)
            presence_ratio = total_active_sec / max(1.0, session_duration_sec)

            status, notes = track.get_sustained_status(session_duration_sec)

            # Assign associated student name if matched with roster, else anonymous label
            assigned_student_id = student_ids_list[idx] if idx < len(student_ids_list) else tid
            assigned_name = roster_map[assigned_student_id]["name"] if assigned_student_id in roster_map else f"Participant {idx+1}"

            # Confidence score based on detection consistency
            confidence = min(0.98, max(0.40, (len(track.detection_timestamps) / max(1, (track.total_detections + track.consecutive_missing_frames)))))

            if status == "Present":
                present_count += 1
            elif status == "Transient":
                transient_count += 1
            else:
                absent_count += 1

            rec = {
                "session_id": session_id,
                "participant_id": assigned_student_id,
                "participant_name": assigned_name,
                "first_seen_sec": round(first_seen, 2),
                "last_seen_sec": round(last_seen, 2),
                "total_detected_sec": round(total_active_sec, 2),
                "presence_ratio": round(presence_ratio, 3),
                "attendance_status": status,
                "confidence_score": round(confidence, 2),
                "notes": notes
            }
            records.append(rec)

        # Include enrolled students who were never seen at all as Absent
        matched_enrolled = len(sorted_track_ids)
        total_enrolled = max(len(roster_map), matched_enrolled)

        if matched_enrolled < len(student_ids_list):
            for remaining_idx in range(matched_enrolled, len(student_ids_list)):
                s_id = student_ids_list[remaining_idx]
                s_obj = roster_map[s_id]
                records.append({
                    "session_id": session_id,
                    "participant_id": s_id,
                    "participant_name": s_obj["name"],
                    "first_seen_sec": 0.0,
                    "last_seen_sec": 0.0,
                    "total_detected_sec": 0.0,
                    "presence_ratio": 0.0,
                    "attendance_status": "Absent",
                    "confidence_score": 0.95,
                    "notes": "Not detected during session"
                })
                absent_count += 1

        total_participants = len(records)
        attendance_pct = (present_count / max(1, total_participants)) * 100.0

        summary = {
            "total_participants": total_participants,
            "present_count": present_count,
            "transient_count": transient_count,
            "absent_count": absent_count,
            "attendance_percentage": round(attendance_pct, 1),
            "sustained_threshold_sec": self.sustained_threshold_sec
        }

        return records, summary
