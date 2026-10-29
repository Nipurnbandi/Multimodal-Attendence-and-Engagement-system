"""
Synthetic Classroom Video & Audio Demo Generator.
Generates a realistic, playable classroom simulation video containing:
- 3 seated students with shifting head poses, eye blinks, and desk movements
- 1 transient attendee who walks in at second 8 and departs at second 12 (to demonstrate false-positive filtering)
- Synthesized speech and silence intervals
- Generates an MP4 with an audio track and processes it through the live multimodal pipeline.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import math
import subprocess
import numpy as np
import cv2
import scipy.io.wavfile as wavfile
from app.config import SAMPLES_DIR, config
from app.database.db import init_db, seed_sample_students

try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_EXE = "ffmpeg"


def draw_realistic_face(
    canvas: np.ndarray,
    center: tuple,
    scale: float = 1.0,
    pose: str = "frontal",
    eyes_open: bool = True
):
    """
    Renders a schematic student face compatible with Haar/feature detectors:
    Draws head oval, eyes, pupils, eyebrows, nose, mouth.
    """
    cx, cy = center
    hw = int(38 * scale)
    hh = int(48 * scale)

    # Face contour
    skin_color = (210, 225, 240)  # BGR
    cv2.ellipse(canvas, (cx, cy), (hw, hh), 0, 0, 360, skin_color, -1)
    cv2.ellipse(canvas, (cx, cy), (hw, hh), 0, 0, 360, (140, 160, 180), 2)

    # Hair
    hair_color = (40, 35, 30)
    cv2.ellipse(canvas, (cx, cy - int(hh * 0.45)), (hw, int(hh * 0.65)), 0, 180, 360, hair_color, -1)

    eye_offset_x = int(14 * scale)
    eye_offset_y = int(6 * scale)

    if pose == "profile":
        eye_offset_x = int(6 * scale)

    # Eyes & Eyebrows
    left_eye_center = (cx - eye_offset_x, cy - eye_offset_y)
    right_eye_center = (cx + eye_offset_x, cy - eye_offset_y)

    if eyes_open:
        # Eye sclera
        cv2.ellipse(canvas, left_eye_center, (int(7 * scale), int(4 * scale)), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(canvas, right_eye_center, (int(7 * scale), int(4 * scale)), 0, 0, 360, (255, 255, 255), -1)
        # Eye pupil
        cv2.circle(canvas, left_eye_center, int(3 * scale), (20, 20, 20), -1)
        cv2.circle(canvas, right_eye_center, int(3 * scale), (20, 20, 20), -1)
        # Eyebrows
        cv2.line(canvas, (left_eye_center[0] - 6, left_eye_center[1] - 6), (left_eye_center[0] + 6, left_eye_center[1] - 7), (40, 35, 30), 2)
        cv2.line(canvas, (right_eye_center[0] - 6, right_eye_center[1] - 7), (right_eye_center[0] + 6, right_eye_center[1] - 6), (40, 35, 30), 2)
    else:
        # Blink
        cv2.line(canvas, (left_eye_center[0] - 6, left_eye_center[1]), (left_eye_center[0] + 6, left_eye_center[1]), (40, 35, 30), 2)
        cv2.line(canvas, (right_eye_center[0] - 6, right_eye_center[1]), (right_eye_center[0] + 6, right_eye_center[1]), (40, 35, 30), 2)

    # Nose
    cv2.line(canvas, (cx, cy - 2), (cx, cy + int(8 * scale)), (140, 160, 180), 2)
    cv2.line(canvas, (cx, cy + int(8 * scale)), (cx + 3, cy + int(8 * scale)), (140, 160, 180), 2)

    # Mouth
    cv2.line(canvas, (cx - int(8 * scale), cy + int(20 * scale)), (cx + int(8 * scale), cy + int(20 * scale)), (70, 70, 190), 2)

    # Shoulders / Torso
    cv2.ellipse(canvas, (cx, cy + hh + 45), (int(hw * 1.8), 40), 0, 180, 360, (160, 100, 50), -1)


def generate_synthetic_audio(duration_sec: float, output_wav: str, sample_rate: int = 16000):
    """
    Generates synthetic classroom audio containing alternating speech and silence intervals.
    """
    total_samples = int(duration_sec * sample_rate)
    audio = np.zeros(total_samples, dtype=np.float32)

    # Simulated speech intervals: [0-7s, 11-18s, 22-28s]
    speech_intervals = [
        (0.5, 7.0),
        (10.5, 17.5),
        (21.0, 27.5)
    ]

    time_vec = np.arange(total_samples) / sample_rate

    # Background room tone / noise floor
    room_noise = np.random.normal(0, 0.003, total_samples).astype(np.float32)
    audio += room_noise

    for start_t, end_t in speech_intervals:
        if start_t >= duration_sec:
            continue
        actual_end = min(duration_sec, end_t)
        mask = (time_vec >= start_t) & (time_vec <= actual_end)

        # Modulated speech formant simulation (fundamental ~160 Hz with harmonics)
        speech_signal = (
            0.15 * np.sin(2 * np.pi * 160 * time_vec) +
            0.10 * np.sin(2 * np.pi * 320 * time_vec) +
            0.07 * np.sin(2 * np.pi * 700 * time_vec) +
            0.04 * np.sin(2 * np.pi * 1800 * time_vec)
        )
        # Amplitude modulation to simulate syllables (3 Hz syllable rate)
        syllables = 0.5 * (1.0 + np.sin(2 * np.pi * 3.5 * time_vec))
        audio[mask] += (speech_signal[mask] * syllables[mask]).astype(np.float32)

    # Clip and save as int16
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    wavfile.write(output_wav, sample_rate, audio_int16)


def create_demo_classroom_video(
    duration_sec: float = 30.0,
    fps: int = 15,
    output_mp4: str = str(SAMPLES_DIR / "classroom_lecture_demo.mp4")
) -> str:
    """
    Synthesizes and muxes an MP4 video showcasing multiple seated students and a transient passerby.
    """
    width, height = 640, 480
    total_frames = int(duration_sec * fps)

    temp_raw_avi = str(SAMPLES_DIR / "temp_raw_video.avi")
    temp_wav = str(SAMPLES_DIR / "temp_audio.wav")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_raw_avi, fourcc, fps, (width, height))

    generate_synthetic_audio(duration_sec, temp_wav)

    # Fixed base positions for 3 seated classroom participants
    desk_positions = [
        {"id": "P1", "x": 130, "y": 240, "scale": 1.1},
        {"id": "P2", "x": 320, "y": 230, "scale": 1.15},
        {"id": "P3", "x": 510, "y": 240, "scale": 1.1}
    ]

    for frame_idx in range(total_frames):
        t = frame_idx / fps
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Classroom background
        canvas[:] = (235, 230, 225)  # Classroom wall

        # Blackboard / Whiteboard
        cv2.rectangle(canvas, (60, 20), (580, 140), (45, 65, 45), -1)
        cv2.rectangle(canvas, (55, 15), (585, 145), (70, 50, 30), 4)
        cv2.putText(canvas, "CS-401: Multimodal Intelligence", (90, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
        cv2.putText(canvas, "Lecture: Attention Dynamics & Vision", (110, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 200), 1)

        # Teacher's Desk / Classroom boundary
        cv2.rectangle(canvas, (0, 380), (640, 480), (190, 180, 170), -1)
        cv2.line(canvas, (0, 380), (640, 380), (140, 130, 120), 3)

        # Draw 3 continuous seated participants with subtle natural motion
        for idx, student in enumerate(desk_positions):
            # Micro-motion: natural breathing and shifting
            offset_x = int(3 * math.sin(t * 1.5 + idx * 2))
            offset_y = int(2 * math.cos(t * 1.8 + idx))

            # Occasional eye blink
            eyes_open = not (int(t * 2) % 7 == 0 and (frame_idx % 6 < 2))

            # Head pose shifts: P3 turns to peer between seconds 14 and 18
            pose = "frontal"
            if idx == 2 and (14.0 <= t <= 18.5):
                pose = "profile"

            center = (student["x"] + offset_x, student["y"] + offset_y)
            draw_realistic_face(canvas, center, scale=student["scale"], pose=pose, eyes_open=eyes_open)

            # Draw desk in front of student
            cv2.rectangle(canvas, (student["x"] - 65, student["y"] + 65), (student["x"] + 65, student["y"] + 115), (140, 95, 60), -1)
            cv2.rectangle(canvas, (student["x"] - 65, student["y"] + 65), (student["x"] + 65, student["y"] + 115), (100, 65, 40), 2)

        # Draw Transient Student: enters at t=8s, walks past, exits at t=12s
        if 8.0 <= t <= 12.0:
            walk_progress = (t - 8.0) / 4.0
            transient_x = int(20 + walk_progress * 600)
            transient_y = 190
            draw_realistic_face(canvas, (transient_x, transient_y), scale=0.85, pose="profile", eyes_open=True)

        # Frame counter & time overlay
        cv2.putText(canvas, f"Time: {t:.1f}s | Frame: {frame_idx}", (15, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)

        out.write(canvas)

    out.release()

    # Mux AVI and WAV into clean MP4 using imageio_ffmpeg
    cmd = [
        FFMPEG_EXE,
        "-y",
        "-i", temp_raw_avi,
        "-i", temp_wav,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        output_mp4
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception:
        # Fallback if x264 isn't available: copy raw video
        if os.path.exists(temp_raw_avi):
            os.replace(temp_raw_avi, output_mp4)

    # Clean up temp files
    if os.path.exists(temp_raw_avi) and os.path.exists(output_mp4) and temp_raw_avi != output_mp4:
        os.remove(temp_raw_avi)
    if os.path.exists(temp_wav):
        os.remove(temp_wav)

    return output_mp4


def seed_database_with_sample_sessions():
    """Seeds the SQLite database with 2 realistic lecture sessions for immediate demonstration."""
    from app.main_pipeline import MultimodalPipelineCoordinator

    init_db()
    seed_sample_students()

    demo_video_path = str(SAMPLES_DIR / "classroom_lecture_demo.mp4")
    if not os.path.exists(demo_video_path):
        create_demo_classroom_video(duration_sec=25.0, output_mp4=demo_video_path)

    coordinator = MultimodalPipelineCoordinator()
    print("Processing sample lecture session through real multimodal pipeline...")
    result = coordinator.run_pipeline(
        video_path=demo_video_path,
        session_title="CS401: Artificial Intelligence & Autonomous Agents",
        course_code="CS401",
        instructor_name="Dr. Aris Thorne",
        source_type="sample_demo"
    )
    print(f"Sample session successfully generated: {result['session_id']}")
    return result


if __name__ == "__main__":
    seed_database_with_sample_sessions()
