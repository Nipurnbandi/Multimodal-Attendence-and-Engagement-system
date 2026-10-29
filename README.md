# Multimodal Attendance & Engagement Agent 🎓

> **An offline-first, privacy-preserving classroom intelligence & decision-support system.**  
> Built with Python, OpenCV, MediaPipe, SciPy, SQLAlchemy, Plotly, and Streamlit.

---

## 🌟 Overview

The **Multimodal Attendance & Engagement Agent** moves beyond manual roll calls and basic biometric turnstiles. It combines:
1. **Computer Vision**: Multi-person spatial tracking, head orientation estimation (frontal alert vs. desk down vs. profile turned), and micro-motion differencing.
2. **Acoustic Signal Processing**: Offline Root-Mean-Square (RMS) energy extraction, Zero-Crossing Rate (ZCR), and Voice Activity Detection (VAD) to segment verbal interaction from lecture monologue.
3. **Sustained Presence Attendance**: Filters momentary passerby false positives by requiring consistent duration thresholds before confirming presence.
4. **Multimodal Behavioral Indicators**: Fuses visual gaze, physical movement, and acoustic density into transparent interval dynamics.
5. **AI Pedagogical Reasoning Agent**: Explains *why* specific pacing lapses or attention drops occurred, providing instructors with formative teaching suggestions (e.g., Think-Pair-Share, posture breaks, discussion synthesis).
6. **Live Webcam Viva Mode**: Allows real-time evaluation directly using a laptop camera.

---

## 🏗️ Architecture

```
                      +-----------------------------+
                      |   Recorded Video / Webcam   |
                      +-----------------------------+
                                     |
                    +----------------+----------------+
                    |                                 |
                    v                                 v
        +-----------------------+         +-----------------------+
        |   Audio Processing    |         |   Computer Vision     |
        |  - RMS Energy & ZCR   |         |  - Frontal/Profile    |
        |  - Voice Activity VAD |         |  - Motion Difference  |
        |  - Silence Ratios     |         |  - Centroid MOT Track |
        +-----------------------+         +-----------------------+
                    |                                 |
                    +----------------+----------------+
                                     |
                                     v
                    +---------------------------------+
                    |    Multimodal Fusion Engine     |
                    |  - Engagement Score E(t)        |
                    |  - 30s/60s Interval Dynamics    |
                    |  - Sustained Presence Filter    |
                    +---------------------------------+
                                     |
                                     v
                    +---------------------------------+
                    |    AI Reasoning Expert Agent    |
                    |  - Attention Drops (>=20 pts)   |
                    |  - Interaction Peaks (>=50% aud)|
                    |  - Pedagogical Suggestions      |
                    +---------------------------------+
                                     |
                                     v
                    +---------------------------------+
                    |  Streamlit Dashboard & Reports  |
                    |  - Interactive Plotly Timeline  |
                    |  - CSV Gradebook & JSON Dumps   |
                    |  - SQLite Persistent Database   |
                    +---------------------------------+
```

---

## 📁 Project Structure

```
multimodal_attendance_agent/
│
├── app/
│   ├── main.py                  # Primary Streamlit web application & dashboard
│   ├── config.py                # System thresholds, weights, and directory paths
│   ├── main_pipeline.py         # End-to-end multimodal pipeline orchestrator
│   │
│   ├── vision/
│   │   ├── detector.py          # Haar cascades, head orientation & motion differencing
│   │   ├── tracker.py           # Centroid-based tracker & sustained presence evaluator
│   │   └── video_processor.py   # Frame sampling pipeline & preview extraction
│   │
│   ├── audio/
│   │   └── processor.py         # RMS energy, ZCR, adaptive noise-floor VAD
│   │
│   ├── attendance/
│   │   └── manager.py           # Roster reconciliation & sustained presence filtering
│   │
│   ├── engagement/
│   │   └── engine.py            # Multimodal fusion formula & interval dynamics
│   │
│   ├── agent/
│   │   └── recommendation_agent.py # Explainable pedagogical reasoning engine
│   │
│   ├── analytics/
│   │   └── reporter.py          # Plotly charts, CSV/JSON export & Markdown reports
│   │
│   └── database/
│       ├── models.py            # SQLAlchemy models (Sessions, Attendance, Metrics)
│       └── db.py                # Database engine, session helpers, privacy purge
│
├── data/
│   ├── generate_sample_data.py  # Synthetic classroom video & lecture seed generator
│   ├── input/                   # Stored uploaded video files
│   ├── processed/               # Extracted WAV audio and cache
│   ├── reports/                 # Exported CSVs and JSON analytics
│   └── samples/                 # Pre-packaged demo lecture MP4
│
├── tests/
│   ├── test_vision.py           # Unit tests for tracker, sustained presence, detector
│   ├── test_audio.py            # Unit tests for RMS, ZCR, VAD segmentation
│   ├── test_engagement.py       # Unit tests for engagement formula & intervals
│   ├── test_agent.py            # Unit tests for pedagogical rule triggers
│   └── test_database.py         # Unit tests for SQLite persistence & privacy deletion
│
├── UNIVERSITY_DOCUMENTATION.md  # Comprehensive academic project report & viva Q&A
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment configuration template
└── README.md                    # Project manual & documentation
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- Python 3.10+ (Fully compatible with Python 3.10, 3.11, 3.12, 3.13, 3.14).
- A webcam (optional, for live viva demonstration).

### 2. Install Dependencies
```bash
# Clone or navigate to the project directory
cd multimodal_attendance_agent

# Install dependencies via pip
pip install -r requirements.txt
```

*(Note: Audio demuxing uses `imageio-ffmpeg` which bundles a standalone zero-config FFmpeg binary; no external system installs required!)*

---

## 🏃 Running the Application

### 1. Launch the Web Dashboard
Run via Python module invocation:
```powershell
python -m streamlit run app/main.py
```
Or double-click the included batch launcher:
```powershell
.\run_dashboard.bat
```
Open your browser at `http://localhost:8501`.

### 2. Generate Demo Data (One-Click)
To test immediately without uploading a custom video:
```bash
python data/generate_sample_data.py
```
Or simply click the **"⚡ Load Demo Sample Lecture"** button in the Streamlit sidebar!

### 3. Run Automated Unit Tests
```bash
python -m pytest tests/ -v
```
All 10 unit and integration tests run in under 2 seconds.

---

## 📊 Key Dashboard Features

1. **Dashboard & Analytics**:
   - Executive metrics ribbon: Attendance %, Average Engagement, Present Count, Filtered Transients.
   - Interactive Plotly multi-line timeline: Estimated Engagement, Frontal Attention Signal, Movement Level, and Audio Activity % with marked attention-drop alerts.
   - Attendance status donut chart (Present vs. Transient vs. Absent).
   - Horizontal bar chart of presence duration per student.
2. **Analyze Classroom Video**:
   - Upload any MP4, AVI, MOV, or MKV classroom lecture.
   - Live step-by-step progress bar (audio demuxing $\rightarrow$ frame processing $\rightarrow$ attendance $\rightarrow$ multimodal fusion $\rightarrow$ AI reasoning).
3. **Live Webcam Demo (Viva Mode)**:
   - Real-time video processing directly from laptop camera.
   - Live bounding boxes, participant track IDs, attention signals, and movement scores.
4. **Attendance Roster**:
   - Filter by status (`Present`, `Transient`, `Absent`).
   - Search by student name or roll number.
   - One-click export to CSV gradebook.
5. **AI Teaching Suggestions**:
   - Expandable recommendation cards showing the **Observed Signal**, the **Pedagogical Rationale (WHY)**, and the **Actionable Recommendation**.
6. **Reports & Exports**:
   - One-click export of structured JSON analytics and formatted Markdown reports.
7. **Privacy & Settings**:
   - Transparent engagement formula documentation.
   - One-click permanent session data deletion.

---

## 🛡️ Privacy and Ethical Standards

- **Assistive, Non-Punitive Design**: The system explicitly provides formative teaching support for the instructor. It does not make autonomous disciplinary or grading decisions.
- **Estimated Behavioral Indicators**: Attention metrics are strictly defined and labeled as physical proxies (head pose, gaze direction, movement), never claiming infallible mind-reading.
- **Local & Offline**: All processing occurs strictly on the host machine. No audio or video is transmitted to third-party cloud APIs.
- **Transient Presence Filtering**: Accidental passerby appearances (<10s) are classified as "Transient" and excluded from attendance.
- **No Biometric Vector Archiving**: Raw video is not permanently retained; only derived numerical time-series metrics are saved.

---

## 🎓 University Viva Presentation Cheat-Sheet

| Topic | Key Answer |
| :--- | :--- |
| **Sustained Presence** | Enforces a dual-threshold: minimum active duration ($\ge 10\text{s}$) and session ratio ($\ge 35\%$). Transient passersby are classified as "Transient" to eliminate false positives. |
| **Multimodal Formula** | $E(t) = 0.40 \cdot \text{Attention} + 0.25 \cdot \text{Movement} + 0.20 \cdot \text{Audio} + 0.15 \cdot \text{Continuity}$. Balanced to prevent penalizing quiet note-taking. |
| **Audio Processing** | Passive RMS energy and Zero-Crossing Rate (ZCR) with an adaptive 20th-percentile noise-floor threshold. Detects speech vs. silence without private speech transcription. |
| **Explainable AI** | A rule-based expert engine that inspects structured intervals and explains the pedagogical reason (*WHY*) behind every teaching recommendation. |
