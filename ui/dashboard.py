"""
ui/dashboard.py
Therapist Dashboard — PyQt5 GUI for the PhysioCV system.

Layout:
  Left panel  — live webcam feed with skeleton overlay
  Center panel — joint angle bars + posture score
  Right panel  — patient selector, exercise picker, session controls

Run standalone:
    python -m ui.dashboard
"""

import sys
import time
import threading
from pathlib import Path
from typing import Optional, Dict

import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QComboBox, QLineEdit, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QProgressBar, QStatusBar, QSizePolicy, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QFrame,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette

from core.session_runner import SessionRunner
from data.session_logger import SessionLogger
from reports.report_generator import ReportGenerator
from analysis.angle_calculator import JointAngle


# ── Color palette ──────────────────────────────────────────────────────────────
CLR_TEAL   = "#1D9E75"
CLR_TEAL_L = "#E1F5EE"
CLR_AMBER  = "#EF9F27"
CLR_CORAL  = "#D85A30"
CLR_BG     = "#F8F7F4"
CLR_CARD   = "#FFFFFF"
CLR_BORDER = "#D3D1C7"
CLR_TEXT   = "#2C2C2A"
CLR_MUTED  = "#5F5E5A"

JOINT_DISPLAY_NAMES = {
    "left_knee":      "Left knee",
    "right_knee":     "Right knee",
    "left_elbow":     "Left elbow",
    "right_elbow":    "Right elbow",
    "left_shoulder":  "Left shoulder",
    "right_shoulder": "Right shoulder",
    "left_hip":       "Left hip",
    "right_hip":      "Right hip",
    "spine":          "Spine",
}

STATUS_COLORS = {
    "good": CLR_TEAL,
    "warn": CLR_AMBER,
    "bad":  CLR_CORAL,
}

EXERCISES = ["standing", "squat", "shoulder_raise", "plank", "front_lunge", "sitting", "lying"]


# ── Worker thread ───────────────────────────────────────────────────────────────
class SessionSignals(QObject):
    frame_ready   = pyqtSignal(np.ndarray, dict, int)
    session_ended = pyqtSignal(int)
    error         = pyqtSignal(str)

class SessionWorker(QThread):
    def __init__(self, patient_id: str, exercise: str, camera: int = 0):
        super().__init__()
        self.patient_id = patient_id
        self.exercise   = exercise
        self.camera     = camera
        self.signals    = SessionSignals()
        self._running   = True
        self._runner: Optional[SessionRunner] = None
        self.session_id: Optional[int] = None   # ← add this
        self.pre_session_id: Optional[int] = None 

    def run(self):
        try:
            self._runner = SessionRunner()
            session_id = self._runner.run(
                patient_id=self.patient_id,
                exercise=self.exercise,
                camera=self.camera,
                on_frame_callback=self._on_frame,
                session_id=self.pre_session_id, 
            )
            self.session_id = session_id          # ← store it
            self.signals.session_ended.emit(session_id)
        except Exception as e:
            self.signals.error.emit(str(e))

    def _on_frame(self, frame, angles, score):
        if self._running:
            self.signals.frame_ready.emit(frame.copy(), dict(angles), score)

    def stop(self):
        self._running = False


# ── Joint angle bar widget ──────────────────────────────────────────────────────
class AngleBar(QWidget):
    """Single joint row: label | progress bar | angle value."""

    def __init__(self, joint_name: str, parent=None):
        super().__init__(parent)
        self.joint_name = joint_name

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)

        self.label = QLabel(JOINT_DISPLAY_NAMES.get(joint_name, joint_name))
        self.label.setFixedWidth(120)
        self.label.setStyleSheet(f"color: {CLR_MUTED}; font-size: 12px;")

        self.bar = QProgressBar()
        self.bar.setRange(0, 180)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(8)
        self.bar.setStyleSheet(self._bar_style(CLR_TEAL))

        self.value_label = QLabel("—")
        self.value_label.setFixedWidth(50)
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.value_label.setStyleSheet(f"color: {CLR_TEXT}; font-size: 12px; font-weight: 600;")

        self.status_dot = QLabel("●")
        self.status_dot.setFixedWidth(14)
        self.status_dot.setStyleSheet(f"color: {CLR_TEAL}; font-size: 10px;")

        layout.addWidget(self.label)
        layout.addWidget(self.bar, 1)
        layout.addWidget(self.value_label)
        layout.addWidget(self.status_dot)

    def _bar_style(self, color: str) -> str:
        return f"""
            QProgressBar {{
                background: {CLR_BORDER};
                border-radius: 4px;
                border: none;
            }}
            QProgressBar::chunk {{
                background: {color};
                border-radius: 4px;
            }}
        """

    def update_joint(self, joint: JointAngle):
        color = STATUS_COLORS.get(joint.status, CLR_TEAL)
        self.bar.setValue(min(180, max(0, int(joint.angle_deg))))
        self.bar.setStyleSheet(self._bar_style(color))
        self.value_label.setText(f"{joint.angle_deg:.0f}°")
        self.status_dot.setStyleSheet(f"color: {color}; font-size: 10px;")


# ── Score ring widget ───────────────────────────────────────────────────────────
class ScoreWidget(QWidget):
    """Displays the overall posture score as a large number with color coding."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(8, 8, 8, 8)

        self.score_label = QLabel("—")
        self.score_label.setAlignment(Qt.AlignCenter)
        self.score_label.setStyleSheet(
            f"font-size: 48px; font-weight: 800; color: {CLR_TEAL};"
        )

        self.sub_label = QLabel("posture score")
        self.sub_label.setAlignment(Qt.AlignCenter)
        self.sub_label.setStyleSheet(f"font-size: 11px; color: {CLR_MUTED}; letter-spacing: 1px;")

        layout.addWidget(self.score_label)
        layout.addWidget(self.sub_label)

    def update_score(self, score: int):
        color = (
            CLR_TEAL  if score >= 80
            else CLR_AMBER if score >= 50
            else CLR_CORAL
        )
        self.score_label.setText(str(score))
        self.score_label.setStyleSheet(
            f"font-size: 48px; font-weight: 800; color: {color};"
        )


# ── Main dashboard window ───────────────────────────────────────────────────────
class TherapistDashboard(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhysioCV — Therapist Dashboard")
        self.resize(1280, 760)
        self.setStyleSheet(f"QMainWindow {{ background: {CLR_BG}; }}")

        self._worker: Optional[SessionWorker] = None
        self._session_id: Optional[int] = None
        self._current_session_id: Optional[int] = None
        self._frame_count = 0
        self._session_start: Optional[float] = None

        self._build_ui()
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_clock)
        self._timer.start(1000)

    # ── UI construction ─────────────────────────────────────────────────────────

    def _card(self, title: str = "") -> QGroupBox:
        box = QGroupBox(title)
        box.setStyleSheet(f"""
            QGroupBox {{
                background: {CLR_CARD};
                border: 0.5px solid {CLR_BORDER};
                border-radius: 10px;
                font-size: 12px;
                font-weight: 600;
                color: {CLR_MUTED};
                padding-top: 20px;
                margin-top: 6px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                top: 4px;
            }}
        """)
        return box

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ── Left: live feed ──────────────────────────────────────────────────
        feed_card = self._card("Live feed")
        feed_layout = QVBoxLayout(feed_card)

        self.feed_label = QLabel()
        self.feed_label.setMinimumSize(640, 480)
        self.feed_label.setAlignment(Qt.AlignCenter)
        self.feed_label.setStyleSheet(
            f"background: #1a1a1a; border-radius: 6px; color: {CLR_MUTED}; font-size: 13px;"
        )
        self.feed_label.setText("Camera feed will appear here\nwhen session starts")
        feed_layout.addWidget(self.feed_label)

        # Feed stats bar
        stats_row = QHBoxLayout()
        self.fps_label   = QLabel("FPS: —")
        self.conf_label  = QLabel("Confidence: —")
        self.clock_label = QLabel("00:00")
        for lbl in [self.fps_label, self.conf_label, self.clock_label]:
            lbl.setStyleSheet(f"font-size: 11px; color: {CLR_MUTED};")
        stats_row.addWidget(self.fps_label)
        stats_row.addStretch()
        stats_row.addWidget(self.conf_label)
        stats_row.addStretch()
        stats_row.addWidget(self.clock_label)
        feed_layout.addLayout(stats_row)

        root.addWidget(feed_card, 5)

        # ── Center: angles + score ───────────────────────────────────────────
        center_col = QVBoxLayout()
        center_col.setSpacing(10)

        score_card = self._card("Posture score")
        score_layout = QVBoxLayout(score_card)
        self.score_widget = ScoreWidget()
        score_layout.addWidget(self.score_widget)
        center_col.addWidget(score_card)

        angles_card = self._card("Joint angles")
        angles_layout = QVBoxLayout(angles_card)
        angles_layout.setSpacing(4)
        self._angle_bars: Dict[str, AngleBar] = {}
        for joint in JOINT_DISPLAY_NAMES:
            bar = AngleBar(joint)
            self._angle_bars[joint] = bar
            angles_layout.addWidget(bar)
        center_col.addWidget(angles_card, 1)

        root.addLayout(center_col, 3)

        # ── Right: controls ──────────────────────────────────────────────────
        right_col = QVBoxLayout()
        right_col.setSpacing(10)

        # Patient panel
        patient_card = self._card("Patient")
        patient_layout = QVBoxLayout(patient_card)

        patient_layout.addWidget(QLabel("Patient ID"))
        self.patient_input = QLineEdit()
        self.patient_input.setPlaceholderText("e.g. P001")
        self.patient_input.setStyleSheet(self._input_style())
        patient_layout.addWidget(self.patient_input)

        patient_layout.addWidget(QLabel("Exercise"))
        self.exercise_combo = QComboBox()
        self.exercise_combo.addItems([e.replace("_", " ").title() for e in EXERCISES])
        self.exercise_combo.setStyleSheet(self._input_style())
        patient_layout.addWidget(self.exercise_combo)

        patient_layout.addWidget(QLabel("Camera index"))
        self.camera_input = QLineEdit("0")
        self.camera_input.setStyleSheet(self._input_style())
        patient_layout.addWidget(self.camera_input)

        right_col.addWidget(patient_card)

        # Session controls
        ctrl_card = self._card("Session")
        ctrl_layout = QVBoxLayout(ctrl_card)

        self.start_btn = QPushButton("▶  Start session")
        self.start_btn.clicked.connect(self._start_session)
        self.start_btn.setStyleSheet(self._btn_style(CLR_TEAL, "#fff"))

        self.stop_btn = QPushButton("■  Stop session")
        self.stop_btn.clicked.connect(self._stop_session)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(self._btn_style(CLR_CORAL, "#fff"))

        ctrl_layout.addWidget(self.start_btn)
        ctrl_layout.addWidget(self.stop_btn)
        # ctrl_layout.addWidget(self.report_btn)
        right_col.addWidget(ctrl_card)

        root.addLayout(right_col, 2)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready — configure patient and press Start")

    def _input_style(self) -> str:
        return f"""
            QLineEdit, QComboBox {{
                border: 0.5px solid {CLR_BORDER};
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 13px;
                background: {CLR_BG};
                color: {CLR_TEXT};
            }}
            QLineEdit:focus, QComboBox:focus {{
                border-color: {CLR_TEAL};
            }}
        """

    def _btn_style(self, bg: str, fg: str) -> str:
        return f"""
            QPushButton {{
                background: {bg};
                color: {fg};
                border: none;
                border-radius: 7px;
                padding: 10px 16px;
                font-size: 13px;
                font-weight: 600;
            }}
            QPushButton:hover {{ opacity: 0.9; }}
            QPushButton:disabled {{
                background: {CLR_BORDER};
                color: {CLR_MUTED};
            }}
        """

    def _start_session(self):
        patient_id = self.patient_input.text().strip() or "Unknown"
        exercise   = EXERCISES[self.exercise_combo.currentIndex()]
        camera     = int(self.camera_input.text().strip() or "0")

        self._session_start = time.time()
        self._frame_count   = 0

        # Pre-create session in DB so we have the ID immediately
        try:
            db_path = Path(__file__).resolve().parent.parent / "data" / "sessions.db"
            logger = SessionLogger(db_path)
            self._current_session_id = logger.start_session(patient_id, exercise)
            logger.close()
            print(f"[DEBUG] Pre-created session_id={self._current_session_id}")
        except Exception as e:
            print(f"[DEBUG] Could not pre-create session: {e}")
        self._current_session_id = None

        self._worker = SessionWorker(patient_id, exercise, camera)
        self._worker.pre_session_id = self._current_session_id   # ← add this
        self._worker.signals.frame_ready.connect(self._on_frame)
        self._worker.signals.session_ended.connect(self._on_session_ended)
        self._worker.signals.error.connect(self._on_error)
        self._worker.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        # self.report_btn.setEnabled(False)
        self.status_bar.showMessage(f"Session running — {exercise} — {patient_id}")


    def _stop_session(self):
        if self._worker:
            self._worker.stop()
            self._worker.terminate()
            self._worker.wait()  # wait for thread to fully finish

            # Manually end session since signal may not have fired
            if self._current_session_id:
                try:
                    db_path = Path(__file__).resolve().parent.parent / "data" / "sessions.db"
                    logger = SessionLogger(db_path)
                    logger.end_session(self._current_session_id)
                    logger.close()
                    self._session_id = self._current_session_id
                except Exception as e:
                    print(f"[DEBUG] end_session error: {e}")

            self._worker = None

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        # self.report_btn.setEnabled(True)
        self.status_bar.showMessage("Session stopped — you can now export the PDF")

    def _on_session_ended(self, session_id: int):
        self._session_id = session_id
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        # self.report_btn.setEnabled(True)
        self.status_bar.showMessage(f"Session {session_id} complete — export PDF when ready")

    def _on_error(self, msg: str):
        self.status_bar.showMessage(f"Error: {msg}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    # ── Frame handler ───────────────────────────────────────────────────────────

    def _on_frame(self, frame: np.ndarray, angles: dict, score: int):
        self._frame_count += 1

        # Convert BGR → RGB → QPixmap
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.feed_label.width(), self.feed_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self.feed_label.setPixmap(pixmap)

        # Update angle bars
        for joint_name, bar in self._angle_bars.items():
            if joint_name in angles:
                bar.update_joint(angles[joint_name])

        # Update score
        self.score_widget.update_score(score)

        # FPS estimate
        elapsed = time.time() - (self._session_start or time.time())
        fps = self._frame_count / elapsed if elapsed > 0 else 0
        self.fps_label.setText(f"FPS: {fps:.1f}")

    # ── Clock ───────────────────────────────────────────────────────────────────

    def _update_clock(self):
        if self._session_start and self._worker and self._worker.isRunning():
            elapsed = int(time.time() - self._session_start)
            m, s = divmod(elapsed, 60)
            self.clock_label.setText(f"{m:02d}:{s:02d}")


# ── Standalone entry ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = TherapistDashboard()
    window.show()
    sys.exit(app.exec_())