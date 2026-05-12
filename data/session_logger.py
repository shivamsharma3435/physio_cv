"""
data/session_logger.py
Records per-frame joint angles, posture scores, and session metadata
to a SQLite database and exports CSV summaries for reporting.
"""

import sqlite3
import csv
import json
import time
from pathlib import Path
from typing import Dict, Optional
from analysis.angle_calculator import JointAngle


DB_PATH = Path("data/sessions.db")


class SessionLogger:
    """Persists session data to SQLite and supports CSV export."""

    def __init__(self, db_path: Path = DB_PATH):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id  TEXT NOT NULL,
                exercise    TEXT NOT NULL,
                started_at  REAL NOT NULL,
                ended_at    REAL,
                avg_score   REAL
            );

            CREATE TABLE IF NOT EXISTS frames (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER REFERENCES sessions(id),
                ts          REAL NOT NULL,
                score       INTEGER,
                angles_json TEXT
            );
        """)
        self.conn.commit()

    def start_session(self, patient_id: str, exercise: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO sessions (patient_id, exercise, started_at) VALUES (?,?,?)",
            (patient_id, exercise, time.time()),
        )
        self.conn.commit()
        return cur.lastrowid

    def log_frame(
        self,
        session_id: int,
        score: int,
        angles: Dict[str, JointAngle],
    ):
        angles_data = {
            name: {"angle": j.angle_deg, "status": j.status, "dev": j.deviation_deg}
            for name, j in angles.items()
        }
        self.conn.execute(
            "INSERT INTO frames (session_id, ts, score, angles_json) VALUES (?,?,?,?)",
            (session_id, time.time(), score, json.dumps(angles_data)),
        )
        self.conn.commit()

    def end_session(self, session_id: int):
        cur = self.conn.execute(
            "SELECT AVG(score) FROM frames WHERE session_id=?", (session_id,)
        )
        avg = cur.fetchone()[0] or 0
        self.conn.execute(
            "UPDATE sessions SET ended_at=?, avg_score=? WHERE id=?",
            (time.time(), round(avg, 1), session_id),
        )
        self.conn.commit()

    def export_csv(self, session_id: int, out_path: Path):
        """Exports all frames of a session to CSV for reporting."""
        rows = self.conn.execute(
            "SELECT ts, score, angles_json FROM frames WHERE session_id=? ORDER BY ts",
            (session_id,),
        ).fetchall()

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "score", "joint", "angle_deg", "status", "deviation"])
            for ts, score, angles_json in rows:
                angles = json.loads(angles_json)
                for joint, data in angles.items():
                    writer.writerow([
                        round(ts, 3), score, joint,
                        data["angle"], data["status"], data["dev"]
                    ])

    def get_session_summary(self, session_id: int) -> dict:
        session = self.conn.execute(
            "SELECT patient_id, exercise, started_at, ended_at, avg_score "
            "FROM sessions WHERE id=?", (session_id,)
        ).fetchone()

        frame_count = self.conn.execute(
            "SELECT COUNT(*) FROM frames WHERE session_id=?", (session_id,)
        ).fetchone()[0]

        bad_frames = self.conn.execute(
            "SELECT COUNT(*) FROM frames WHERE session_id=? AND score < 50",
            (session_id,)
        ).fetchone()[0]

        return {
            "session_id":  session_id,
            "patient_id":  session[0],
            "exercise":    session[1],
            "started_at":  session[2],
            "ended_at":    session[3],
            "avg_score":   session[4],
            "frame_count": frame_count,
            "bad_frames":  bad_frames,
            "duration_s":  round((session[3] or time.time()) - session[2], 1),
        }

    def close(self):
        self.conn.close()
