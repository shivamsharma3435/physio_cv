"""
Physiotherapy Posture Monitoring System
Entry point — launches the therapist dashboard.
"""

import sys
import os
import cv2
from PyQt5.QtWidgets import QApplication
from ui.dashboard import TherapistDashboard

os.environ["GLOG_minloglevel"] = "3"


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("PhysioCV — Posture Monitor")
    window = TherapistDashboard()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
