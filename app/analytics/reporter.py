"""
Analytics, Reporting, and Visualization Engine.
Generates interactive Plotly figures, CSV/JSON data exports, and comprehensive markdown/HTML session reports.
"""

from typing import Dict, List, Any, Optional
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from app.config import config, REPORTS_DIR

class SessionAnalyticsReporter:
    """
    Produces visual dashboards and exportable summaries for classroom sessions.
    """

    @staticmethod
    def create_engagement_timeline_chart(interval_metrics: List[Dict[str, Any]]) -> go.Figure:
        """
        Creates an interactive multi-line chart tracking Engagement Score,
        Visual Attention Signal, Movement Activity, and Audio Activity across intervals.
        """
        if not interval_metrics:
            fig = go.Figure()
            fig.update_layout(title="No interval data available")
            return fig

        df = pd.DataFrame(interval_metrics)
        labels = df["timestamp_label"].tolist()

        fig = go.Figure()

        # Engagement Score
        fig.add_trace(go.Scatter(
            x=labels,
            y=df["avg_engagement_score"],
            mode="lines+markers",
            name="Estimated Engagement",
            line=dict(color="#2563EB", width=3),
            marker=dict(size=7)
        ))

        # Attention Signal
        fig.add_trace(go.Scatter(
            x=labels,
            y=df["avg_attention_signal"],
            mode="lines",
            name="Frontal Attention Signal",
            line=dict(color="#10B981", width=2, dash="dash")
        ))

        # Movement Activity
        fig.add_trace(go.Scatter(
            x=labels,
            y=df["avg_movement_activity"],
            mode="lines",
            name="Movement Level",
            line=dict(color="#F59E0B", width=2, dash="dot")
        ))

        # Audio Activity (scaled to 100)
        audio_pct = df["audio_activity_ratio"] * 100.0
        fig.add_trace(go.Bar(
            x=labels,
            y=audio_pct,
            name="Audio Activity %",
            marker=dict(color="rgba(168, 85, 247, 0.35)")
        ))

        # Mark attention drops
        drop_indices = [i for i, m in enumerate(interval_metrics) if m.get("attention_drop_flag", False)]
        if drop_indices:
            fig.add_trace(go.Scatter(
                x=[labels[i] for i in drop_indices],
                y=[df["avg_engagement_score"].iloc[i] for i in drop_indices],
                mode="markers",
                name="Attention Drop Alert",
                marker=dict(color="#EF4444", size=13, symbol="triangle-down")
            ))

        fig.update_layout(
            title="Classroom Dynamics & Engagement Timeline",
            xaxis_title="Time Window (mm:ss)",
            yaxis_title="Indicator Score (0 - 100)",
            yaxis=dict(range=[0, 105]),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            margin=dict(l=40, r=40, t=60, b=40)
        )
        return fig

    @staticmethod
    def create_attendance_distribution_chart(summary: Dict[str, Any]) -> go.Figure:
        """
        Donut chart illustrating attendance status breakdown: Present, Transient, Absent.
        """
        labels = ["Present (Sustained)", "Transient (Filtered)", "Absent"]
        values = [
            summary.get("present_count", 0),
            summary.get("transient_count", 0),
            summary.get("absent_count", 0)
        ]
        colors = ["#10B981", "#F59E0B", "#EF4444"]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker=dict(colors=colors),
            textinfo="label+value+percent",
            hoverinfo="label+value"
        )])

        fig.update_layout(
            title="Attendance Status Breakdown",
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        return fig

    @staticmethod
    def create_participant_duration_chart(records: List[Dict[str, Any]]) -> go.Figure:
        """
        Horizontal bar chart comparing active detected duration per participant.
        """
        if not records:
            return go.Figure()

        df = pd.DataFrame(records).sort_values("total_detected_sec", ascending=True)

        colors = []
        for s in df["attendance_status"]:
            if s == "Present":
                colors.append("#10B981")
            elif s == "Transient":
                colors.append("#F59E0B")
            else:
                colors.append("#EF4444")

        fig = go.Figure(go.Bar(
            x=df["total_detected_sec"],
            y=df["participant_name"],
            orientation="h",
            marker=dict(color=colors),
            text=[f"{sec:.1f}s ({stat})" for sec, stat in zip(df["total_detected_sec"], df["attendance_status"])],
            textposition="auto"
        ))

        fig.update_layout(
            title="Presence Duration by Participant (Seconds)",
            xaxis_title="Active Detected Time (seconds)",
            yaxis_title="Participant",
            template="plotly_white",
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return fig

    @staticmethod
    def export_attendance_csv(session_id: str, records: List[Dict[str, Any]]) -> str:
        """Exports attendance records to a CSV file and returns the file path."""
        df = pd.DataFrame(records)
        target_path = str(REPORTS_DIR / f"attendance_{session_id}.csv")
        df.to_csv(target_path, index=False)
        return target_path

    @staticmethod
    def export_full_analytics_json(session_data: Dict[str, Any]) -> str:
        """Exports comprehensive session analytics to JSON and returns file path."""
        session_id = session_data.get("session_id", "session")
        target_path = str(REPORTS_DIR / f"analytics_{session_id}.json")
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, default=str)
        return target_path

    @staticmethod
    def generate_markdown_report(
        session_meta: Dict[str, Any],
        attendance_summary: Dict[str, Any],
        engagement_summary: Dict[str, Any],
        recommendations: List[Dict[str, Any]]
    ) -> str:
        """Generates a structured markdown session report for university documentation and instructor review."""
        md = []
        md.append(f"# Classroom Intelligence Report: {session_meta.get('title', 'Lecture Session')}")
        md.append(f"**Session ID:** `{session_meta.get('id', 'N/A')}` | **Course:** {session_meta.get('course_code', 'UNIV-101')} | **Instructor:** {session_meta.get('instructor_name', 'Faculty')}")
        md.append(f"**Date:** {session_meta.get('created_at', 'Today')} | **Duration:** {session_meta.get('duration_seconds', 0):.1f}s")
        md.append("\n---\n")

        md.append("## 1. Executive Summary")
        md.append(f"- **Overall Estimated Engagement:** `{engagement_summary.get('overall_avg_engagement', 0):.1f} / 100`")
        md.append(f"- **Attendance Rate:** `{attendance_summary.get('attendance_percentage', 0):.1f}%` ({attendance_summary.get('present_count', 0)} Present, {attendance_summary.get('transient_count', 0)} Transient, {attendance_summary.get('absent_count', 0)} Absent)")
        md.append(f"- **Peak Engagement Interval:** `{engagement_summary.get('peak_engagement', 0):.1f}` | **Lowest Interval:** `{engagement_summary.get('lowest_engagement', 0):.1f}`")
        md.append(f"- **Attention Drops Flagged:** {len(engagement_summary.get('attention_drops', []))}")
        md.append(f"- **Interaction Peaks Flagged:** {len(engagement_summary.get('interaction_peaks', []))}")
        md.append("\n---\n")

        md.append("## 2. AI Pedagogical Recommendations & Feedback")
        for idx, r in enumerate(recommendations, 1):
            sev_badge = f"**[{r.get('severity', 'info').upper()}]**"
            md.append(f"### {idx}. {r.get('title', 'Suggestion')} {sev_badge}")
            md.append(f"- **Category:** {r.get('category', 'General')} | **Window:** `{r.get('timestamp_range', 'Session')}`")
            md.append(f"- **Observation:** {r.get('observation')}")
            md.append(f"- **Pedagogical Rationale (WHY):** {r.get('reasoning')}")
            md.append(f"- **Actionable Recommendation:** {r.get('suggestion')}\n")

        md.append("---\n")
        md.append("## 3. Privacy and Methodological Disclaimer")
        md.append("> *This report was compiled by the Multimodal Attendance & Engagement Agent. All metrics are non-punitive estimated behavioral indicators designed strictly for instructional decision support. No biometric facial recognition databases were stored.*")

        return "\n".join(md)
