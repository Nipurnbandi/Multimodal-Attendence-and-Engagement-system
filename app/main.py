"""
Multimodal Attendance & Engagement Agent - Web Application.
Interactive Streamlit Dashboard providing:
1. Video Upload & Live Pipeline Execution
2. Real-time Webcam Viva Demo
3. Executive Intelligence Dashboard
4. Interactive Engagement & Dynamics Timeline Charts
5. Attendance Records & Roster Reconciliation
6. AI-Driven Teaching Suggestions with Pedagogical Rationales
7. CSV, JSON, and Markdown Report Exports
8. Privacy-Preserving Session Management
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import cv2
import numpy as np

from app.config import config, INPUT_DIR, SAMPLES_DIR, REPORTS_DIR
from app.database.db import init_db, seed_sample_students, get_db, delete_session_data
from app.database.models import Session, AttendanceRecord, IntervalMetric, SessionRecommendation
from app.main_pipeline import MultimodalPipelineCoordinator
from app.analytics.reporter import SessionAnalyticsReporter
from app.vision.video_processor import VideoProcessor

# Initialize Database Schema & Students on first run
init_db()
seed_sample_students()

# Page Configuration
st.set_page_config(
    page_title="Multimodal Attendance & Engagement Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern university-grade dashboard styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-present {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-transient {
        background-color: #FEF08A;
        color: #713F12;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-absent {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .suggestion-box {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 14px;
    }
</style>
""", unsafe_allow_html=True)


def get_all_sessions():
    """Retrieve all stored sessions ordered by creation date."""
    with get_db() as db:
        sessions = db.query(Session).order_by(Session.created_at.desc()).all()
        return [s.to_dict() for s in sessions]


def get_session_details(session_id: str):
    """Retrieve full data hierarchy for a selected session."""
    with get_db() as db:
        session_obj = db.query(Session).filter_by(id=session_id).first()
        if not session_obj:
            return None
        attendance = [a.to_dict() for a in session_obj.attendance_records]
        intervals = [i.to_dict() for i in session_obj.interval_metrics]
        recommendations = [r.to_dict() for r in session_obj.recommendations]
        return {
            "meta": session_obj.to_dict(),
            "attendance": attendance,
            "intervals": intervals,
            "recommendations": recommendations
        }


# Sidebar Navigation & Settings
st.sidebar.image("https://img.icons8.com/fluency/96/classroom.png", width=64)
st.sidebar.title("Classroom AI")
st.sidebar.markdown("**Decision-Support Agent**")

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 Dashboard & Analytics",
        "📹 Analyze Classroom Video",
        "🎥 Live Webcam Demo",
        "📋 Attendance Roster",
        "💡 AI Teaching Suggestions",
        "📁 Reports & Exports",
        "⚙️ System Settings"
    ]
)

st.sidebar.markdown("---")
# Session Selector
available_sessions = get_all_sessions()
session_options = {s["id"]: f"{s['title']} ({s['id']})" for s in available_sessions}

if available_sessions:
    selected_session_id = st.sidebar.selectbox(
        "Select Active Session:",
        options=list(session_options.keys()),
        format_func=lambda x: session_options[x]
    )
else:
    selected_session_id = None
    st.sidebar.info("No sessions in database. Analyze a video or load demo sample.")

# Quick Demo Loader
if st.sidebar.button("⚡ Load Demo Sample Lecture", use_container_width=True):
    from data.generate_sample_data import seed_database_with_sample_sessions
    with st.spinner("Generating and analyzing simulated lecture session..."):
        res = seed_database_with_sample_sessions()
        st.sidebar.success(f"Loaded: {res['session_id']}")
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("🔒 **Privacy-Preserving**: All computer vision, acoustic signals, and AI reasoning execute 100% locally on your machine.")


# ---------------------------------------------------------
# VIEW 1: Dashboard & Analytics
# ---------------------------------------------------------
if nav_choice == "📊 Dashboard & Analytics":
    st.markdown('<div class="main-title">Multimodal Attendance & Engagement Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Privacy-preserving classroom intelligence: from presence detection to AI-guided pedagogical support.</div>', unsafe_allow_html=True)

    if not selected_session_id:
        st.warning("Please upload a classroom video or click '⚡ Load Demo Sample Lecture' in the sidebar to view analytics.")
    else:
        data = get_session_details(selected_session_id)
        if not data:
            st.error("Session data could not be retrieved.")
        else:
            meta = data["meta"]
            attendance = data["attendance"]
            intervals = data["intervals"]
            recs = data["recommendations"]

            # Key Metrics Ribbon
            col1, col2, col3, col4, col5 = st.columns(5)
            
            present_count = sum(1 for a in attendance if a["attendance_status"] == "Present")
            transient_count = sum(1 for a in attendance if a["attendance_status"] == "Transient")
            absent_count = sum(1 for a in attendance if a["attendance_status"] == "Absent")
            total_students = len(attendance)

            col1.metric("Course / Session", meta.get("course_code", "CS401"), meta.get("title", "Lecture")[:18] + "...")
            col2.metric("Attendance Rate", f"{meta.get('attendance_percentage', 0):.1f}%", f"{present_count} of {total_students} Present")
            col3.metric("Estimated Engagement", f"{meta.get('average_engagement', 0):.1f}/100", "Behavioral Indicator")
            col4.metric("Filtered Transients", transient_count, "False-Positive Protection")
            col5.metric("Session Duration", f"{meta.get('duration_seconds', 0):.1f}s", f"{len(intervals)} Time Intervals")

            st.markdown("---")

            # Main Charts
            c1, c2 = st.columns([2, 1])

            with c1:
                st.subheader("📈 Multimodal Engagement & Dynamics Timeline")
                fig_timeline = SessionAnalyticsReporter.create_engagement_timeline_chart(intervals)
                st.plotly_chart(fig_timeline, use_container_width=True)

            with c2:
                st.subheader("👥 Attendance Status")
                att_summary = {
                    "present_count": present_count,
                    "transient_count": transient_count,
                    "absent_count": absent_count
                }
                fig_donut = SessionAnalyticsReporter.create_attendance_distribution_chart(att_summary)
                st.plotly_chart(fig_donut, use_container_width=True)

            # Secondary Analytics
            st.markdown("---")
            c3, c4 = st.columns([1, 1])

            with c3:
                st.subheader("⏱️ Presence Duration by Participant")
                fig_dur = SessionAnalyticsReporter.create_participant_duration_chart(attendance)
                st.plotly_chart(fig_dur, use_container_width=True)

            with c4:
                st.subheader("💡 Top Pedagogical Suggestions")
                if not recs:
                    st.info("No alerts or suggestions flagged for this session.")
                else:
                    for r in recs[:3]:
                        sev_color = "#EF4444" if r["severity"] == "warning" else ("#F59E0B" if r["severity"] == "action_suggested" else "#2563EB")
                        st.markdown(f"""
                        <div class="suggestion-box" style="border-left: 4px solid {sev_color};">
                            <h4 style="margin: 0 0 6px 0; color: #1F2937;">{r['title']}</h4>
                            <p style="font-size: 0.9rem; color: #4B5563; margin-bottom: 4px;"><strong>Observed:</strong> {r['observation']}</p>
                            <p style="font-size: 0.9rem; color: #1E40AF; margin: 0;"><strong>Suggestion:</strong> {r['suggestion']}</p>
                        </div>
                        """, unsafe_allow_html=True)


# ---------------------------------------------------------
# VIEW 2: Analyze Classroom Video
# ---------------------------------------------------------
elif nav_choice == "📹 Analyze Classroom Video":
    st.markdown('<div class="main-title">Analyze Classroom Video</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload a recorded lecture (MP4, AVI, MOV) for automated multimodal analysis.</div>', unsafe_allow_html=True)

    with st.form("video_upload_form"):
        col1, col2 = st.columns(2)
        with col1:
            session_title = st.text_input("Session Title", value="CS401: Distributed Systems & Agents")
            course_code = st.text_input("Course Code", value="CS-401")
        with col2:
            instructor_name = st.text_input("Instructor Name", value="Dr. A. Thorne")
            uploaded_file = st.file_uploader("Upload Classroom Video File", type=["mp4", "avi", "mov", "mkv"])

        submit_btn = st.form_submit_button("🚀 Start Multimodal Analysis", use_container_width=True)

    if submit_btn:
        if not uploaded_file:
            st.error("Please select a video file to upload.")
        else:
            # Save uploaded video to data/input
            saved_path = str(INPUT_DIR / uploaded_file.name)
            with open(saved_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"Video uploaded successfully: `{uploaded_file.name}` ({uploaded_file.size / 1024 / 1024:.1f} MB)")

            progress_bar = st.progress(0.0)
            status_text = st.empty()

            coordinator = MultimodalPipelineCoordinator()

            def update_progress(pct, msg):
                progress_bar.progress(pct)
                status_text.markdown(f"**Status:** {msg}")

            try:
                result = coordinator.run_pipeline(
                    video_path=saved_path,
                    session_title=session_title,
                    course_code=course_code,
                    instructor_name=instructor_name,
                    source_type="uploaded_video",
                    progress_callback=update_progress
                )
                st.balloons()
                st.success(f"Analysis complete! Session ID: `{result['session_id']}`")
                time.sleep(1.5)
                st.rerun()
            except Exception as e:
                st.error(f"Error during video processing: {str(e)}")


# ---------------------------------------------------------
# VIEW 3: Live Webcam Demo (Viva Mode)
# ---------------------------------------------------------
elif nav_choice == "🎥 Live Webcam Demo":
    st.markdown('<div class="main-title">Live Webcam Demo (Viva Mode)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Test real-time participant tracking, sustained presence calculation, and live attention indicators directly using your laptop camera.</div>', unsafe_allow_html=True)

    st.info("💡 **Viva Tip:** Click 'Start Webcam' to show real-time face detection, head orientation classification (frontal vs turned away), and sustained presence accumulation.")

    run_cam = st.toggle("🔴 Activate Camera Stream", value=False)
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()

    if run_cam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access camera. Please verify camera permissions.")
        else:
            processor = VideoProcessor(sample_fps=4.0)
            start_time = time.time()

            try:
                while run_cam:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    current_sec = time.time() - start_time
                    annotated_frame, metrics = processor.process_single_frame(frame, current_sec)

                    # Convert BGR to RGB for Streamlit display
                    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                    active_count = metrics["active_participants"]
                    att_score = metrics["avg_attention_signal"]
                    mot_score = metrics["avg_movement_score"]

                    stats_placeholder.markdown(f"""
                    **Live Diagnostics (Elapsed: {current_sec:.1f}s):**
                    - **Detected Participants:** `{active_count}`
                    - **Attention Signal:** `{att_score:.1f} / 100` (Frontal Gaze / Orientation)
                    - **Movement Activity:** `{mot_score:.1f} / 100` (Micro-motion difference)
                    """)

                    time.sleep(0.08)
            finally:
                cap.release()


# ---------------------------------------------------------
# VIEW 4: Attendance Roster
# ---------------------------------------------------------
elif nav_choice == "📋 Attendance Roster":
    st.markdown('<div class="main-title">Classroom Attendance Roster</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Verified sustained presence records reconciled against the enrolled class roster.</div>', unsafe_allow_html=True)

    if not selected_session_id:
        st.warning("Please select or load a session from the sidebar.")
    else:
        data = get_session_details(selected_session_id)
        if data:
            records = data["attendance"]
            meta = data["meta"]

            # Filter controls
            c1, c2 = st.columns([1, 2])
            with c1:
                status_filter = st.multiselect(
                    "Filter by Status:",
                    options=["Present", "Transient", "Absent"],
                    default=["Present", "Transient", "Absent"]
                )
            with c2:
                search_query = st.text_input("Search by Name or Student ID:", "")

            filtered = [
                r for r in records
                if r["attendance_status"] in status_filter and
                (search_query.lower() in r["participant_name"].lower() or search_query.lower() in r["participant_id"].lower())
            ]

            # Display table
            display_rows = []
            for r in filtered:
                display_rows.append({
                    "Student ID": r["participant_id"],
                    "Participant Name": r["participant_name"],
                    "Status": r["attendance_status"],
                    "First Seen (s)": f"{r['first_seen_sec']:.1f}",
                    "Last Seen (s)": f"{r['last_seen_sec']:.1f}",
                    "Active Duration": f"{r['total_detected_sec']:.1f}s",
                    "Presence %": f"{r['presence_ratio']*100:.1f}%",
                    "Confidence": f"{r['confidence_score']*100:.0f}%",
                    "Audit Notes": r["notes"] or ""
                })

            df_display = pd.DataFrame(display_rows)
            st.dataframe(df_display, use_container_width=True, height=380)

            # CSV Download
            csv_data = df_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Attendance as CSV",
                data=csv_data,
                file_name=f"attendance_{meta['id']}.csv",
                mime="text/csv"
            )


# ---------------------------------------------------------
# VIEW 5: AI Teaching Suggestions
# ---------------------------------------------------------
elif nav_choice == "💡 AI Teaching Suggestions":
    st.markdown('<div class="main-title">AI Teaching Suggestions</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Explainable decision-support recommendations grounded in multimodal classroom dynamics.</div>', unsafe_allow_html=True)

    if not selected_session_id:
        st.warning("Please select or load a session from the sidebar.")
    else:
        data = get_session_details(selected_session_id)
        if data:
            recs = data["recommendations"]
            if not recs:
                st.info("No specific teaching interventions suggested for this session.")
            else:
                for idx, r in enumerate(recs, 1):
                    severity = r.get("severity", "info")
                    badge_color = "#DC2626" if severity == "warning" else ("#D97706" if severity == "action_suggested" else "#2563EB")

                    with st.expander(f"{idx}. {r['title']} — [{r['category']}]", expanded=True):
                        st.markdown(f"**Time Window:** `{r.get('timestamp_range', 'Session-Wide')}` | **Priority:** <span style='color: {badge_color}; font-weight: 700;'>{severity.upper()}</span>", unsafe_allow_html=True)
                        st.markdown(f"**🔍 Observed Multimodal Signal:**\n{r['observation']}")
                        st.markdown(f"**🧠 Pedagogical Rationale (WHY):**\n{r['reasoning']}")
                        st.markdown(f"**✅ Actionable Recommendation:**\n{r['suggestion']}")


# ---------------------------------------------------------
# VIEW 6: Reports & Exports
# ---------------------------------------------------------
elif nav_choice == "📁 Reports & Exports":
    st.markdown('<div class="main-title">Reports & Data Exports</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Generate university submission reports, audit files, and export session analytics.</div>', unsafe_allow_html=True)

    if not selected_session_id:
        st.warning("Please select a session.")
    else:
        data = get_session_details(selected_session_id)
        if data:
            meta = data["meta"]
            attendance = data["attendance"]
            intervals = data["intervals"]
            recs = data["recommendations"]

            att_summary = {
                "attendance_percentage": meta.get("attendance_percentage", 0.0),
                "present_count": sum(1 for a in attendance if a["attendance_status"] == "Present"),
                "transient_count": sum(1 for a in attendance if a["attendance_status"] == "Transient"),
                "absent_count": sum(1 for a in attendance if a["attendance_status"] == "Absent")
            }
            eng_summary = {
                "overall_avg_engagement": meta.get("average_engagement", 0.0),
                "peak_engagement": max([i["avg_engagement_score"] for i in intervals]) if intervals else 0.0,
                "lowest_engagement": min([i["avg_engagement_score"] for i in intervals]) if intervals else 0.0,
                "attention_drops": [i for i in intervals if i.get("attention_drop_flag")],
                "interaction_peaks": [i for i in intervals if i.get("interaction_peak_flag")]
            }

            md_content = SessionAnalyticsReporter.generate_markdown_report(
                session_meta=meta,
                attendance_summary=att_summary,
                engagement_summary=eng_summary,
                recommendations=recs
            )

            st.text_area("Preview Markdown Report", md_content, height=280)

            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "📥 Download Markdown Report (.md)",
                    data=md_content,
                    file_name=f"report_{meta['id']}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            with c2:
                import json
                json_data = json.dumps(data, indent=2, default=str)
                st.download_button(
                    "📥 Download Complete Analytics (.json)",
                    data=json_data,
                    file_name=f"analytics_{meta['id']}.json",
                    mime="application/json",
                    use_container_width=True
                )


# ---------------------------------------------------------
# VIEW 7: System Settings & Privacy Controls
# ---------------------------------------------------------
elif nav_choice == "⚙️ System Settings":
    st.markdown('<div class="main-title">System Settings & Privacy Management</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Configure thresholds, review mathematical formulas, and manage data privacy.</div>', unsafe_allow_html=True)

    st.subheader("📐 Multimodal Engagement Formula")
    st.latex(r"E(t) = w_{\text{gaze}} \cdot \text{Attention}(t) + w_{\text{motion}} \cdot \text{Movement}(t) + w_{\text{audio}} \cdot \text{Audio}(t) + w_{\text{presence}} \cdot \text{Continuity}(t)")
    st.markdown(f"""
    - **Frontal Attention Weight ($w_1$):** `{config.weight_frontal_attention}`
    - **Movement Activity Weight ($w_2$):** `{config.weight_movement_activity}`
    - **Audio Activity Weight ($w_3$):** `{config.weight_audio_activity}`
    - **Presence Continuity Weight ($w_4$):** `{config.weight_presence_continuity}`
    - **Sustained Presence Threshold:** `{config.sustained_presence_threshold_sec} seconds`
    """)

    st.markdown("---")
    st.subheader("🛡️ Privacy & Compliance Controls")
    st.markdown("""
    Under academic privacy standards:
    - Raw classroom videos are processed locally in volatile memory and are not stored permanently.
    - No facial biometric templates or identity embeddings are persisted.
    - All session analytics can be purged permanently below.
    """)

    if selected_session_id:
        if st.button(f"🗑️ Permanently Delete Session Data: `{selected_session_id}`", type="primary"):
            success = delete_session_data(selected_session_id)
            if success:
                st.success(f"Session {selected_session_id} and derived records purged.")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Failed to delete session.")
