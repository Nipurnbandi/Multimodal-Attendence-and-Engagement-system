# University Project Academic Documentation Report

**Project Title:** Multimodal Attendance & Engagement Agent: An Explainable Classroom Intelligence & Decision-Support System  
**Academic Degree:** Bachelor of Technology / Master of Technology in Computer Science & Engineering / Artificial Intelligence  
**Domain:** Computer Vision, Digital Signal Processing, Explainable AI (XAI), Human-Centered Decision Support  

---

## 1. Abstract

Traditional classroom attendance recording mechanisms rely either on manual roll calls—which disrupt lecture momentum and remain vulnerable to proxy sign-ins—or on basic biometric scanners that only record punctual presence at a doorway without offering pedagogical insight into student attentiveness, participation dynamics, or instructor pacing effectiveness. 

This project presents the **Multimodal Attendance & Engagement Agent**, an offline-first, privacy-preserving classroom intelligence platform that bridges spatial attendance verification with explainable engagement dynamics. The system synchronizes lightweight computer vision (multi-participant tracking, head orientation estimation, and inter-frame motion differencing) with passive acoustic signal processing (Root-Mean-Square energy calculation and Voice Activity Detection) to generate continuous interval dynamics. 

To eliminate false-positive attendance marks caused by transient passersby, the system enforces a strict *sustained presence* threshold. Crucially, rather than claiming to read minds or evaluate individual character, the platform models attention as transparent behavioral indicators (e.g. frontal orientation, micro-movement, verbal participation density) and feeds these structured telemetry streams into an AI pedagogical reasoning agent. The agent produces actionable, formative teaching suggestions explaining *why* an intervention is recommended. The entire architecture runs locally on consumer-grade laptop hardware, operates completely offline without paid cloud APIs, and eliminates biometric surveillance risks by discarding raw media and retaining only anonymized statistical indicators.

---

## 2. Problem Statement

Modern higher education and corporate training face a critical feedback deficit:
1. **Punctual vs. Sustained Attendance:** Conventional RFID or biometric scanners mark a student "present" upon physical entry, failing to capture premature departures or momentary passerby false positives.
2. **Pedagogical Blind Spots in Large Lectures:** In lecture halls with 50 to 200 learners, instructors cannot continuously observe subtle student cues (e.g. widespread head drop, sudden restlessness, or 30-minute monologue fatigue).
3. **Black-Box Surveillance Concerns:** Emerging surveillance systems often attempt invasive facial emotion recognition or proprietary "attention scoring," raising grave ethical, legal, and privacy concerns regarding biased algorithmic discipline.
4. **Cloud & Resource Dependencies:** Most contemporary AI systems rely on expensive, latency-prone cloud APIs, rendering them unusable in institutions with limited bandwidth or strict student data residency regulations.

---

## 3. Objectives

- **Automated Sustained Attendance:** Detect and track classroom occupants across video frames, eliminating single-frame appearances through temporal thresholding.
- **Multimodal Behavioral Indicators:** Fuse spatial orientation, physical activity levels, and classroom acoustic interaction into an explainable engagement score.
- **Dynamic Session Discretization:** Partition sessions into configurable time intervals (e.g., 30s or 60s) to identify attention drops, monologue periods, and collaborative interaction peaks.
- **Assistive AI Decision Support:** Implement a rule-based expert reasoning engine that translates telemetry into pedagogical recommendations, providing instructors with the underlying *WHY*.
- **Privacy Preservation & Ethical Design:** Execute 100% offline, avoid permanent raw video storage, and use non-punitive, assistive terminology.
- **Accessible University Demonstration:** Provide a complete web dashboard with interactive charts, CSV/JSON reporting, and a live webcam viva mode.

---

## 4. Existing System vs. Proposed System

| Dimension | Existing Systems (RFID / Face Rec / Manual) | Proposed Multimodal Intelligence Agent |
| :--- | :--- | :--- |
| **Attendance Paradigm** | Point-in-time snapshot (doorway or roll call) | Continuous sustained presence over session duration |
| **False-Positive Mitigation**| Absent (passerby or quick swipe qualifies) | Sustained presence filter (flags transient occupants) |
| **Engagement Insights** | None; presence assumed equal to learning | Continuous multimodal behavioral indicators |
| **Audio Analysis** | None or invasive speech transcription | Non-invasive acoustic signal dynamics (energy & VAD) |
| **Actionable Feedback** | Raw attendance percentages only | AI pedagogical suggestions with explicit rationale |
| **Privacy Footprint** | Biometric facial databases / external cloud storage | 100% local processing; zero biometric template storage |
| **Cost & Connectivity** | Proprietary scanners; paid cloud API subscriptions | Zero-cost open-source stack; fully offline functional |

---

## 5. Scope

### In-Scope
- Ingestion of pre-recorded classroom lectures (MP4, AVI, MOV) and live laptop webcam feeds.
- Real-time multi-person detection, centroid tracking, and track lifetime management.
- Head orientation classification (frontal gaze, downward desk gaze, profile head turns).
- Audio signal processing (energy extraction, noise floor adaptation, speech vs. silence).
- Interval-based aggregation, attention drop detection ($\ge 20$ point drop), and interaction peak detection.
- Explainable AI recommendation generation for pedagogical decision support.
- Streamlit interactive interface, CSV gradebook export, and JSON analytics dumps.

### Out-of-Scope (Design Constraints)
- Direct psychological mind-reading or cognitive EEG tracking (ethically unvalidated).
- Autonomous disciplinary or grading actions against students.
- Permanent archiving of raw high-resolution classroom surveillance footage.

---

## 6. System Architecture

```
+---------------------------------------------------------------------------------------+
|                                    INPUT LAYER                                        |
|             Uploaded Video (MP4/AVI)         OR         Live Webcam Stream            |
+---------------------------------------------------------------------------------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
+---------------------------------------+     +-----------------------------------------+
|        AUDIO PROCESSING MODULE        |     |         VISION PROCESSING MODULE        |
| - Audio Demuxing (imageio_ffmpeg)     |     | - Frame Sampler (2-4 FPS)               |
| - RMS Energy Calculation              |     | - Haar Cascades (Frontal, Profile, Eye) |
| - Zero-Crossing Rate (ZCR)            |     | - Inter-Frame Crop Differencing (Motion)|
| - Adaptive Noise-Floor VAD            |     | - Centroid Distance MOT Tracker         |
| - Speech/Silence Segment Extraction   |     | - Sustained Presence Accumulator        |
+---------------------------------------+     +-----------------------------------------+
                    |                                             |
                    +----------------------+----------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                             MULTIMODAL FUSION ENGINE                                  |
| - Instantaneous Score: E(t) = w1*Attention + w2*Motion + w3*Audio + w4*Presence       |
| - Session Discretization (30s / 60s Slices)                                           |
| - Dynamic Anomaly Flags (Attention Drop >= 20pts, Interaction Peaks >= 50% Audio)     |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                            AI PEDAGOGICAL REASONING AGENT                             |
| - Rule Heuristic Engine: Monologue, Fatigue, Pacing & Discussion Analysis             |
| - Explainable Decision Support: [OBSERVATION] -> [REASON (WHY)] -> [SUGGESTION]       |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                          PERSISTENCE & PRESENTATION LAYER                             |
| - Local SQLite Database (Sessions, AttendanceRecords, IntervalMetrics, Suggestions)   |
| - Streamlit Dashboard (Timeline, Donut Charts, Roster Table, Webcam Viva Mode)        |
| - Export Engine (CSV Gradebook, JSON Analytics, Markdown Academic Report)             |
+---------------------------------------------------------------------------------------+
```

---

## 7. Mathematical Formulations & Algorithms

### 7.1 Acoustic Voice Activity Detection (VAD)
1. **Root Mean Square (RMS) Energy:**
   $$\text{RMS}_w = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2}$$
   Where $x_i$ represents PCM audio samples in window $w$ of duration $0.5\,\text{s}$ at $16\,\text{kHz}$ ($N = 8000$).

2. **Zero-Crossing Rate (ZCR):**
   $$\text{ZCR}_w = \frac{1}{2(N-1)}\sum_{i=1}^{N-1} |\text{sgn}(x_{i+1}) - \text{sgn}(x_i)|$$

3. **Adaptive Activity Threshold:**
   $$T_{\text{energy}} = \max\left(0.005, \theta_{20}(\text{RMS}) \times 1.3\right)$$
   Where $\theta_{20}(\text{RMS})$ is the 20th percentile ambient noise floor. A window is marked active if $\text{RMS}_w > T_{\text{energy}}$ and $0.015 < \text{ZCR}_w < 0.45$.

### 7.2 Multi-Participant Spatial Tracking
Participants are tracked across frames by minimizing Euclidean centroid distance:
$$D(c_i, c_j) = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$$
Matches are accepted if $D < \tau_{\text{dist}}$ ($90\,\text{pixels}$). Tracks missing for greater than $\tau_{\text{missing}}$ ($15\,\text{frames}$) are deregistered.

### 7.3 Inter-Frame Micro-Movement Estimation
Within the bounding box $B = (x, y, w, h)$ of participant $p$:
$$\Delta_{\text{motion}} = \frac{1}{w \cdot h}\sum_{u=0}^{w-1}\sum_{v=0}^{h-1} |I_t(x+u, y+v) - I_{t-1}(x+u, y+v)|$$
$$\text{Score}_{\text{motion}} = \min\left(100.0, \frac{\Delta_{\text{motion}}}{25.0} \times 100.0\right)$$

### 7.4 Multimodal Engagement Score
$$\text{Engagement}(t) = w_1 \cdot A(t) + w_2 \cdot M_{\text{norm}}(t) + w_3 \cdot S_{\text{audio}}(t) + w_4 \cdot P_{\text{ratio}}(t)$$
- $w_1 = 0.40$ (Frontal Attention Indicator $A(t) \in [0, 100]$: alert frontal $= 88$, downward desk $= 65$, profile turned $= 42$).
- $w_2 = 0.25$ (Normalized Movement $M_{\text{norm}}(t)$: penalizes both complete immobility $<5$ and excessive agitation $>85$).
- $w_3 = 0.20$ (Acoustic Activity $S_{\text{audio}}(t)$: $85$ during verbal exchanges, $45$ baseline).
- $w_4 = 0.15$ (Participant Presence Continuity $P_{\text{ratio}}(t) \in [0, 100]$).

### 7.5 Sustained Presence Attendance Filter
$$\text{Status} = \begin{cases} 
\text{Present}, & \text{if } t_{\text{active}} \ge \min(10.0, 0.3 \cdot T_{\text{session}}) \text{ and } \frac{t_{\text{active}}}{T_{\text{session}}} \ge 0.35 \\
\text{Transient}, & \text{if } 0 < t_{\text{active}} < 10.0 \\
\text{Absent}, & \text{if } t_{\text{active}} = 0 
\end{cases}$$

---

## 8. Database Design

```
+-----------------------------------+       +-----------------------------------------+
|             SESSIONS              |       |           ATTENDANCE_RECORDS            |
+-----------------------------------+       +-----------------------------------------+
| id (PK, String)                   |1     *| id (PK, Integer, AutoInc)               |
| title (String)                    |<----->| session_id (FK -> SESSIONS.id)          |
| course_code (String)              |       | participant_id (String)                 |
| instructor_name (String)          |       | participant_name (String)               |
| source_type (String)              |       | first_seen_sec (Float)                  |
| duration_seconds (Float)          |       | last_seen_sec (Float)                   |
| average_engagement (Float)        |       | total_detected_sec (Float)              |
| attendance_percentage (Float)     |       | presence_ratio (Float)                  |
| created_at (DateTime)             |       | attendance_status (Present/Trans/Absent)|
+-----------------------------------+       | confidence_score (Float)                |
         | 1                                | notes (Text)                            |
         |                                  +-----------------------------------------+
         | 1
         |--------+
         |        | *
         |        +------------------------->+-----------------------------------------+
         |                                  |            INTERVAL_METRICS             |
         |                                  +-----------------------------------------+
         |                                  | id (PK, Integer, AutoInc)               |
         |                                  | session_id (FK -> SESSIONS.id)          |
         |                                  | interval_index (Integer)                |
         |                                  | start_sec, end_sec (Float)              |
         |                                  | active_participants (Integer)           |
         |                                  | avg_engagement_score (Float)            |
         |                                  | avg_attention_signal (Float)            |
         |                                  | avg_movement_activity (Float)           |
         |                                  | audio_activity_ratio (Float)            |
         |                                  | attention_drop_flag (Boolean)           |
         |                                  | interaction_peak_flag (Boolean)         |
         |                                  +-----------------------------------------+
         | *
         +--------------------------------->+-----------------------------------------+
                                            |         SESSION_RECOMMENDATIONS         |
                                            +-----------------------------------------+
                                            | id (PK, Integer, AutoInc)               |
                                            | session_id (FK -> SESSIONS.id)          |
                                            | category (String)                       |
                                            | timestamp_range (String)                |
                                            | severity (info/warning/action_suggested)|
                                            | title (String)                          |
                                            | observation (Text)                      |
                                            | reasoning (Text)                        |
                                            | suggestion (Text)                       |
                                            +-----------------------------------------+
```

---

## 9. Hardware & Software Requirements

### Hardware Requirements
- **Processor:** Intel Core i3 / i5 / i7 (or AMD Ryzen 3/5) dual-core or higher.
- **RAM:** Minimum 4 GB RAM (8 GB recommended).
- **Storage:** 500 MB free space for code, cascades, and SQLite storage.
- **Camera:** Standard integrated 720p/1080p laptop webcam (for live viva demonstration).
- **Audio:** Standard integrated laptop microphone or recorded video soundtrack.

### Software Requirements
- **Operating System:** Windows 10/11, macOS, or Linux.
- **Python Version:** Python 3.10+ (tested and verified on Python 3.14).
- **Core Libraries:**
  - `opencv-python`, `opencv-contrib-python` (Computer Vision)
  - `scipy`, `numpy` (Signal Processing)
  - `soundfile`, `imageio-ffmpeg` (Audio Demuxing & Decoding)
  - `streamlit` (Web User Interface)
  - `plotly` (Interactive Analytics Visualizations)
  - `SQLAlchemy` (Object Relational Mapping & SQLite Engine)
  - `pytest` (Automated Verification)

---

## 10. Privacy & Ethical Considerations

1. **Non-Biometric Recognition:** The system tracks anonymous spatial participants (`Participant_01`, `Participant_02`) and performs deterministic roster reconciliation without compiling irreversible facial biometric vectors.
2. **Local Execution:** No classroom media is transmitted to third-party cloud servers or proprietary foundation model APIs.
3. **Volatile Ingestion:** Uploaded videos are analyzed in local volatile storage and can be deleted immediately post-processing.
4. **Assistive Decision Support:** The system explicitly avoids autonomous penalties. Attendance records provide auditable presence timestamps, and engagement flags serve strictly to aid instructional self-reflection.
5. **Non-Punitive Telemetry:** Scores are labeled as *estimated behavioral indicators* rather than infallible judgments of student motivation.

---

## 11. Viva Questions & Model Answers

### Q1: How does your system prevent false-positive attendance from momentary passersby?
**Model Answer:**  
"Standard single-frame presence systems flag any detected face as present, which causes false positives if someone walks past the classroom door. Our system implements a *sustained presence filter*. A participant track must satisfy two criteria: first, a minimum duration threshold ($t_{\text{active}} \ge 10\,\text{seconds}$), and second, a session ratio threshold ($\ge 35\%$ of total lecture duration). Anyone below these thresholds is classified as 'Transient' with an explanatory audit note and excluded from official attendance."

### Q2: Can computer vision truly measure whether a student is paying attention?
**Model Answer:**  
"No, and claiming so would be scientifically and ethically dishonest. Computer vision can only measure physical behavioral proxies: head orientation, gaze direction relative to the board or desk, and physical micro-movement. Our agent explicitly models these as *estimated behavioral indicators*. They are used strictly for aggregate session pacing feedback (e.g. noticing that 70% of the class looked away after 25 minutes of continuous monologue) rather than evaluating individual cognitive states."

### Q3: Why did you choose an offline rule-based expert agent over an external LLM API?
**Model Answer:**  
"First, for privacy and institutional compliance: educational videos and student metadata must not be transmitted to commercial cloud APIs without explicit student consent. Second, for dependability and explainability: rule heuristics guarantee deterministic pedagogical suggestions (e.g., triggering Think-Pair-Share when audio is below 25%) with zero latency, zero cloud costs, and 100% offline reliability during grading or viva evaluation."

### Q4: How does the audio module function without transcribing private student speech?
**Model Answer:**  
"The audio module uses signal-level digital signal processing (DSP) rather than automated speech recognition (ASR). It extracts Root-Mean-Square (RMS) energy and Zero-Crossing Rate (ZCR) across 0.5-second windows. Using an adaptive noise-floor threshold, it detects the presence of verbal interaction versus silence without storing spoken words or creating acoustic voiceprints."

---

## 12. Conclusion & Future Scope

### Conclusion
The Multimodal Attendance & Engagement Agent provides a practical, robust, and privacy-conscious solution for university classroom analytics. By combining sustained presence tracking with explainable multimodal indicators, it empowers instructors with actionable teaching insights while upholding ethical computing standards.

### Future Scope
- Integration with institutional Learning Management Systems (LMS) via LTI standards (e.g. Canvas, Moodle).
- Adaptive camera calibration to support wide-angle pan-tilt-zoom (PTZ) classroom hardware.
- Real-time subtle visual pacing alerts for instructors during live lectures.
