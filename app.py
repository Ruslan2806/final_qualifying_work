import os
import sys
from pathlib import Path

os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False

import json
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QComboBox, QStackedWidget, QMessageBox, QProgressBar)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO

BASE_PATH        = Path(__file__).resolve().parent
CALIBRATION_PATH = BASE_PATH / "camera_calibration" / "calibration.json"
MODEL_PATH       = BASE_PATH / "neural_networks" / "yolov8" / "yolov8n.pt"
# = BASE_PATH / "neural_networks" / "yolov8" / "weights" / "yolov8_best.pt"
# = BASE_PATH / "neural_networks" / "yolov8" / "yolov8n.pt"


def process_frame(frame: np.ndarray, model, calib_data: dict) -> np.ndarray:
    fh        = calib_data["fh"]
    y_horizon = calib_data["y_horizon"]

    results = model.predict(frame, conf=0.4, verbose=False, device=0)[0]

    for box in results.boxes:
        if int(box.cls[0]) != 0:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        y_feet = y2

        print(f"y_feet={y_feet}, y_horizon={y_horizon:.1f}, diff={y_feet - y_horizon:.1f}")

        if y_feet > y_horizon:
            distance = fh / (y_feet - y_horizon)
            dist_text = f"{distance:.1f}m"
        else:
            dist_text = "?"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Pedestrian {dist_text}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame

class VideoWorker(QThread):
    progress_changed = pyqtSignal(int)
    finished         = pyqtSignal(str)
    error            = pyqtSignal(str)

    def __init__(self, input_path: str, output_path: str, model, calib_data: dict):
        super().__init__()
        self.input_path  = input_path
        self.output_path = output_path
        self.model       = model
        self.calib_data  = calib_data

    def run(self):
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            self.error.emit("Не удалось открыть видеофайл")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        processed = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(process_frame(frame, self.model, self.calib_data))
            processed += 1
            if total_frames > 0:
                self.progress_changed.emit(int(processed / total_frames * 100))

        cap.release()
        out.release()
        self.finished.emit(self.output_path)


class CameraWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, camera_index: int, model, calib_data: dict):
        super().__init__()
        self.camera_index = camera_index
        self.model        = model
        self.calib_data   = calib_data
        self._running     = False

    def run(self):
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return
        self._running = True
        while self._running:
            ret, frame = cap.read()
            if ret:
                self.frame_ready.emit(process_frame(frame, self.model, self.calib_data))
        cap.release()

    def stop(self):
        self._running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pedestrian Detection System")
        self.setGeometry(100, 100, 860, 640)

        self.calib_data  = None
        self.model       = None
        self.cam_thread  = None
        self.video_thread = None

        if not self._load_resources():
            sys.exit()

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self._build_menu()
        self._build_video_screen()
        self._build_camera_screen()
        self.stack.setCurrentIndex(0)

    def _load_resources(self) -> bool:
        if not CALIBRATION_PATH.exists():
            QMessageBox.critical(
                self, "Calibration missing",
                f"calibration.json not found:\n{CALIBRATION_PATH}\n\n"
                "Run camera_calibration.py first."
            )
            return False

        if not MODEL_PATH.exists():
            QMessageBox.critical(
                self, "Model missing",
                f"Model not found:\n{MODEL_PATH}"
            )
            return False

        try:
            with open(CALIBRATION_PATH, "r") as f:
                self.calib_data = json.load(f)
            self.model = YOLO(str(MODEL_PATH))
            return True
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            return False

    def _build_menu(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(20)

        title = QLabel("Pedestrian Detection System")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 26px; font-weight: bold;")
        lay.addWidget(title)

        sub = QLabel("Select operating mode")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("font-size: 14px; color: #aaa;")
        lay.addWidget(sub)

        btn_video = QPushButton("📹  Video file processing")
        btn_video.setFixedSize(320, 60)
        btn_video.setStyleSheet(self._btn("#2980b9"))
        btn_video.clicked.connect(lambda: self.stack.setCurrentIndex(1))

        btn_cam = QPushButton("📷  Real-time camera")
        btn_cam.setFixedSize(320, 60)
        btn_cam.setStyleSheet(self._btn("#27ae60"))
        btn_cam.clicked.connect(lambda: self.stack.setCurrentIndex(2))

        lay.addWidget(btn_video, alignment=Qt.AlignCenter)
        lay.addWidget(btn_cam,   alignment=Qt.AlignCenter)

        self.stack.addWidget(w)

    def _build_video_screen(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(12)
        lay.setContentsMargins(20, 20, 20, 20)

        top = QHBoxLayout()
        btn_back = QPushButton("← Back")
        btn_back.setFixedWidth(90)
        btn_back.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        top.addWidget(btn_back)
        top.addWidget(QLabel("<h2>Video Processing</h2>"))
        top.addStretch()
        lay.addLayout(top)

        row_in = QHBoxLayout()
        self.lbl_in = QLabel("No file selected")
        self.lbl_in.setStyleSheet("color: #aaa;")
        btn_in = QPushButton("Select input video")
        btn_in.clicked.connect(self._select_input)
        row_in.addWidget(btn_in)
        row_in.addWidget(self.lbl_in, stretch=1)
        lay.addLayout(row_in)

        row_out = QHBoxLayout()
        self.lbl_out = QLabel("No output path selected")
        self.lbl_out.setStyleSheet("color: #aaa;")
        btn_out = QPushButton("Select output path")
        btn_out.clicked.connect(self._select_output)
        row_out.addWidget(btn_out)
        row_out.addWidget(self.lbl_out, stretch=1)
        lay.addLayout(row_out)

        self.btn_process = QPushButton("▶  Start processing")
        self.btn_process.setFixedHeight(48)
        self.btn_process.setStyleSheet(self._btn("#27ae60"))
        self.btn_process.clicked.connect(self._start_video)
        lay.addWidget(self.btn_process)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        lay.addWidget(self.progress)

        lay.addStretch()
        self.stack.addWidget(w)

    def _build_camera_screen(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(10)
        lay.setContentsMargins(20, 20, 20, 20)

        top = QHBoxLayout()
        btn_back = QPushButton("← Back")
        btn_back.setFixedWidth(90)
        btn_back.clicked.connect(self._stop_cam_and_back)
        top.addWidget(btn_back)
        top.addWidget(QLabel("<h2>Real-time Detection</h2>"))
        top.addStretch()
        lay.addLayout(top)

        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Camera:"))
        self.combo_cam = QComboBox()
        self._detect_cameras()
        cam_row.addWidget(self.combo_cam)
        self.btn_toggle = QPushButton("▶  Start")
        self.btn_toggle.setStyleSheet(self._btn("#27ae60"))
        self.btn_toggle.clicked.connect(self._toggle_camera)
        cam_row.addWidget(self.btn_toggle)
        cam_row.addStretch()
        lay.addLayout(cam_row)

        self.lbl_feed = QLabel("Camera feed will appear here")
        self.lbl_feed.setAlignment(Qt.AlignCenter)
        self.lbl_feed.setStyleSheet("background: #111; color: #555;")
        self.lbl_feed.setMinimumSize(640, 480)
        lay.addWidget(self.lbl_feed, stretch=1)

        self.stack.addWidget(w)

    def _select_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select video", "", "Video (*.mp4 *.avi *.mov)"
        )
        if path:
            self.lbl_in.setText(path)
            self.lbl_in.setStyleSheet("color: white;")

    def _select_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save output as", "output.mp4", "Video (*.mp4)"
        )
        if path:
            self.lbl_out.setText(path)
            self.lbl_out.setStyleSheet("color: white;")

    def _start_video(self):
        in_p  = self.lbl_in.text()
        out_p = self.lbl_out.text()

        if not os.path.exists(in_p):
            QMessageBox.warning(self, "Warning", "Select a valid input video first.")
            return
        if out_p in ("No output path selected", ""):
            QMessageBox.warning(self, "Warning", "Select an output path first.")
            return

        self.btn_process.setEnabled(False)
        self.btn_process.setText("Processing…")
        self.progress.setValue(0)

        self.video_thread = VideoWorker(in_p, out_p, self.model, self.calib_data)
        self.video_thread.progress_changed.connect(self.progress.setValue)
        self.video_thread.finished.connect(self._on_video_done)
        self.video_thread.error.connect(self._on_video_error)
        self.video_thread.start()

    def _on_video_done(self, path: str):
        self.btn_process.setEnabled(True)
        self.btn_process.setText("▶  Start processing")
        self.progress.setValue(100)
        QMessageBox.information(self, "Done", f"Saved to:\n{path}")

    def _on_video_error(self, msg: str):
        self.btn_process.setEnabled(True)
        self.btn_process.setText("▶  Start processing")
        QMessageBox.critical(self, "Error", msg)

    def _detect_cameras(self):
        self.combo_cam.clear()
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                self.combo_cam.addItem(f"Camera {i}", i)
                cap.release()
        if self.combo_cam.count() == 0:
            self.combo_cam.addItem("No cameras found", -1)

    def _toggle_camera(self):
        if self.cam_thread and self.cam_thread.isRunning():
            self.cam_thread.stop()
            self.cam_thread = None
            self.btn_toggle.setText("▶  Start")
            self.btn_toggle.setStyleSheet(self._btn("#27ae60"))
            self.lbl_feed.setText("Camera stopped")
        else:
            idx = self.combo_cam.currentData()
            if idx == -1:
                QMessageBox.warning(self, "Warning", "No camera available.")
                return
            self.cam_thread = CameraWorker(idx, self.model, self.calib_data)
            self.cam_thread.frame_ready.connect(self._update_feed)
            self.cam_thread.start()
            self.btn_toggle.setText("⏹  Stop")
            self.btn_toggle.setStyleSheet(self._btn("#c0392b"))

    def _update_feed(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.lbl_feed.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.lbl_feed.width(), self.lbl_feed.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def _stop_cam_and_back(self):
        if self.cam_thread and self.cam_thread.isRunning():
            self.cam_thread.stop()
            self.cam_thread = None
            self.btn_toggle.setText("▶  Start")
            self.btn_toggle.setStyleSheet(self._btn("#27ae60"))
        self.stack.setCurrentIndex(0)

    def closeEvent(self, event):
        if self.cam_thread and self.cam_thread.isRunning():
            self.cam_thread.stop()
        event.accept()

    @staticmethod
    def _btn(color: str) -> str:
        return (
            f"QPushButton {{"
            f"  background-color: {color}; color: white;"
            f"  border-radius: 6px; font-size: 14px;"
            f"  font-weight: bold; padding: 6px 14px;"
            f"}}"
            f"QPushButton:disabled {{ background-color: #444; color: #888; }}"
            f"QPushButton:hover:enabled {{ background-color: {color}cc; }}"
        )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    from PyQt5.QtGui import QPalette, QColor
    pal = QPalette()
    pal.setColor(QPalette.Window,        QColor(45, 45, 45))
    pal.setColor(QPalette.WindowText,    QColor(220, 220, 220))
    pal.setColor(QPalette.Base,          QColor(30, 30, 30))
    pal.setColor(QPalette.Text,          QColor(220, 220, 220))
    pal.setColor(QPalette.Button,        QColor(60, 60, 60))
    pal.setColor(QPalette.ButtonText,    QColor(220, 220, 220))
    pal.setColor(QPalette.Highlight,     QColor(80, 80, 200))
    pal.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(pal)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())