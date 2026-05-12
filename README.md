# PhysioCV — Patient Posture Monitoring System

A real-time computer vision system to monitor and correct patient posture
during physiotherapy sessions using classical CV + MediaPipe pose estimation.

## Project Structure

```
physio_cv/
├── main.py                        # Entry point (launches dashboard)
├── requirements.txt
├── core/
│   ├── preprocessor.py            # Stage 1: Frame capture & preprocessing
│   ├── edge_detector.py           # Stage 2: Canny + Sobel edge detection
│   ├── contour_analyzer.py        # Stage 3: Body silhouette + convex hull
│   └── session_runner.py          # Orchestrates full pipeline per session
├── analysis/
│   ├── pose_estimator.py          # Stage 4: MediaPipe BlazePose + Kalman
│   ├── angle_calculator.py        # Stage 5a: Joint angle math + scoring
│   └── feedback_engine.py         # Stage 5b: Visual + audio corrections
├── data/
│   └── session_logger.py          # SQLite session recording + CSV export
├── reports/
│   └── report_generator.py        # Automated PDF session reports
└── ui/
    └── dashboard.py               # PyQt5 therapist GUI
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start (headless, no GUI)

```python
from core.session_runner import SessionRunner

runner = SessionRunner()
session_id = runner.run(
    patient_id="P001",
    exercise="squat",   # or "standing", "shoulder_raise"
    camera=0,           # webcam index or path to video file
    show_debug=False,
)
```

## Supported Exercises

| Key              | Description                |
|------------------|----------------------------|
| `standing`       | Standing posture check     |
| `squat`          | Squat form analysis        |
| `shoulder_raise` | Shoulder elevation therapy |

## CV Pipeline Flow

```
Camera frame
  → Preprocessing (grayscale, CLAHE, MOG2 background subtraction)
  → Edge detection (auto Canny + Sobel)
  → Contour analysis (body silhouette, convex hull)
  → Pose estimation (MediaPipe BlazePose, 17 keypoints)
  → Kalman smoothing (per-joint)
  → Angle calculation (dot-product, exercise thresholds)
  → Posture classification (good / warn / bad per joint)
  → Real-time feedback (on-screen text + audio beep)
  → Session logging (SQLite, CSV export)
  → PDF report generation (ReportLab)
```

## Generating a Report

```python
from reports.report_generator import ReportGenerator
from pathlib import Path

rg = ReportGenerator()
rg.generate(session_id=1, out_path=Path("report_P001.pdf"))
rg.close()
```
