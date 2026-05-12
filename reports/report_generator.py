"""
reports/report_generator.py
Generates automated PDF session reports with score timeline,
worst joints table, and posture snapshots using ReportLab.
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.graphics.shapes import Drawing, Line, PolyLine, String, Rect
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics import renderPDF


TEAL  = colors.HexColor("#1D9E75")
CORAL = colors.HexColor("#D85A30")
AMBER = colors.HexColor("#EF9F27")
GRAY  = colors.HexColor("#888780")


class ReportGenerator:
    """Creates a polished A4 PDF report for a completed physiotherapy session."""

    def __init__(self, db_path: Path = Path("data/sessions.db")):
        self.conn = sqlite3.connect(str(db_path))
        self.styles = getSampleStyleSheet()

    def generate(
        self,
        session_id: int,
        out_path: Path,
        snapshot_path: Optional[Path] = None,
    ):
        doc = SimpleDocTemplate(
            str(out_path),
            pagesize=A4,
            topMargin=2*cm, bottomMargin=2*cm,
            leftMargin=2*cm, rightMargin=2*cm,
        )

        story = []
        summary = self._get_summary(session_id)
        frames   = self._get_frames(session_id)

        story += self._header(summary)
        story += self._summary_table(summary)
        story.append(Spacer(1, 0.4*cm))
        story += self._score_chart(frames)
        story.append(Spacer(1, 0.4*cm))
        story += self._joint_table(frames)

        if snapshot_path and Path(snapshot_path).exists():
            story.append(Spacer(1, 0.4*cm))
            story.append(Image(str(snapshot_path), width=8*cm, height=6*cm))

        doc.build(story)

    def _header(self, s: dict) -> list:
        title_style = ParagraphStyle(
            "title", fontSize=18, textColor=TEAL, spaceAfter=4
        )
        sub_style = ParagraphStyle(
            "sub", fontSize=10, textColor=GRAY, spaceAfter=12
        )
        import datetime
        dt = datetime.datetime.fromtimestamp(s["started_at"]).strftime("%d %B %Y, %H:%M")
        return [
            Paragraph("PhysioCV — Session Report", title_style),
            Paragraph(
                f"Patient: <b>{s['patient_id']}</b> &nbsp;|&nbsp; "
                f"Exercise: <b>{s['exercise']}</b> &nbsp;|&nbsp; {dt}",
                sub_style
            ),
        ]

    def _summary_table(self, s: dict) -> list:
        score = s["avg_score"] or 0
        score_color = TEAL if score >= 80 else (AMBER if score >= 50 else CORAL)
        data = [
            ["Metric", "Value"],
            ["Average posture score", f"{score:.1f} / 100"],
            ["Session duration", f"{s['duration_s']:.0f} s"],
            ["Frames analysed", str(s["frame_count"])],
            ["Frames with poor posture (<50)", str(s["bad_frames"])],
        ]
        t = Table(data, colWidths=[8*cm, 8*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), TEAL),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTSIZE",   (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1EFE8")]),
            ("TEXTCOLOR",  (1, 1), (1, 1), score_color),
            ("FONTNAME",   (1, 1), (1, 1), "Helvetica-Bold"),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#D3D1C7")),
        ]))
        return [t]

    def _score_chart(self, frames: list) -> list:
        if not frames:
            return [Paragraph("No frame data available.", self.styles["Normal"])]

        scores = [f[1] for f in frames]
        n = len(scores)
        step = max(1, n // 100)
        sampled = scores[::step]
        data = [[(i, v) for i, v in enumerate(sampled)]]

        d = Drawing(400, 120)
        lp = LinePlot()
        lp.x = 30; lp.y = 10; lp.width = 360; lp.height = 100
        lp.data = data
        lp.lines[0].strokeColor = TEAL
        lp.lines[0].strokeWidth = 1.5
        lp.xValueAxis.valueMin = 0
        lp.xValueAxis.valueMax = len(sampled)
        lp.yValueAxis.valueMin = 0
        lp.yValueAxis.valueMax = 100
        d.add(lp)

        title = ParagraphStyle("ct", fontSize=9, textColor=GRAY)
        return [
            Paragraph("Posture score over time", title),
            d,
        ]

    def _joint_table(self, frames: list) -> list:
        joint_stats: dict = {}
        for _, _, angles_json in frames:
            if not angles_json:
                continue
            for joint, data in json.loads(angles_json).items():
                if joint not in joint_stats:
                    joint_stats[joint] = {"angles": [], "bad": 0, "warn": 0}
                joint_stats[joint]["angles"].append(data["angle"])
                if data["status"] == "bad":
                    joint_stats[joint]["bad"] += 1
                elif data["status"] == "warn":
                    joint_stats[joint]["warn"] += 1

        rows = [["Joint", "Mean angle °", "Warn frames", "Bad frames"]]
        for joint, st in sorted(joint_stats.items()):
            mean_a = np.mean(st["angles"]) if st["angles"] else 0
            rows.append([
                joint.replace("_", " ").title(),
                f"{mean_a:.1f}",
                str(st["warn"]),
                str(st["bad"]),
            ])

        t = Table(rows, colWidths=[5*cm, 4*cm, 3*cm, 3*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), TEAL),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1EFE8")]),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#D3D1C7")),
        ]))
        title = ParagraphStyle("ct", fontSize=9, textColor=GRAY)
        return [Paragraph("Joint-by-joint breakdown", title), Spacer(1, 0.2*cm), t]

    def _get_summary(self, session_id: int) -> dict:
        row = self.conn.execute(
            "SELECT patient_id, exercise, started_at, ended_at, avg_score "
            "FROM sessions WHERE id=?", (session_id,)
        ).fetchone()
        frame_count = self.conn.execute(
            "SELECT COUNT(*) FROM frames WHERE session_id=?", (session_id,)
        ).fetchone()[0]
        bad = self.conn.execute(
            "SELECT COUNT(*) FROM frames WHERE session_id=? AND score < 50",
            (session_id,)
        ).fetchone()[0]
        import time
        return {
            "session_id": session_id, "patient_id": row[0], "exercise": row[1],
            "started_at": row[2], "ended_at": row[3] or time.time(),
            "avg_score": row[4], "frame_count": frame_count, "bad_frames": bad,
            "duration_s": round((row[3] or time.time()) - row[2], 1),
        }

    def _get_frames(self, session_id: int) -> list:
        return self.conn.execute(
            "SELECT ts, score, angles_json FROM frames WHERE session_id=? ORDER BY ts",
            (session_id,)
        ).fetchall()

    def close(self):
        self.conn.close()
