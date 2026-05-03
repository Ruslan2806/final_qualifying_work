import os
import sys
import subprocess
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
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor
from ultralytics import YOLO

BASE_PATH        = Path(__file__).resolve().parent
CALIBRATION_PATH = BASE_PATH / "camera_calibration" / "calibration.json"
MODEL_PATH      = BASE_PATH / "neural_networks" / "yolov8" / "yolov8n.pt"
# = BASE_PATH / "neural_networks" / "yolov8" / "weights" / "yolov8_best.pt"

foot_history    = {}
prev_distances  = {}
speed_state     = {}
trajectories    = {}
kalman_states   = {}
kalman_covs     = {}
smooth_foot_x   = {}
smooth_foot_y   = {}
danger_levels   = {}

TTC_CRITICAL  = 3.0   # секунды — критическая опасность
TTC_WARNING   = 6.0   # секунды — предупреждение  
DIST_MIN      = 1.0   # метр — минимальная безопасная дистанция

def process_frame(frame: np.ndarray, model, calib_data: dict, fps: float) -> np.ndarray:
    import numpy as np
    import cv2

    global kalman_states, kalman_covs
    global prev_distances, speed_state
    global trajectories, smooth_foot_x, smooth_foot_y
    global danger_levels

    # очищаем уровни опасности на каждый кадр
    danger_levels.clear()

    fh        = calib_data["fh"]
    y_horizon = calib_data["y_horizon"]

    frame_h = frame.shape[0]
    calib_h = calib_data.get("image_height", frame_h)
    scale   = frame_h / calib_h if calib_h > 0 else 1.0

    fh_s = fh * scale
    yh_s = y_horizon * scale

    results = model.track(
        frame,
        conf=0.4,
        verbose=False,
        device=0,
        tracker="bytetrack.yaml",
        persist=True,
        amp=False
    )[0]

    if results.boxes is None:
        return frame

    dt       = 1.0 / max(fps, 1e-5)
    alpha    = 0.2
    sigma_a  = 0.5
    deadband = 0.03

    # ── Kalman ─────────────────────────────────────────────────────────
    F = np.array([[1, dt],
                  [0, 1]], dtype=np.float32)

    H = np.array([[1, 0]], dtype=np.float32)

    Q = np.array([
        [dt**4 / 4, dt**3 / 2],
        [dt**3 / 2, dt**2]
    ], dtype=np.float32) * sigma_a**2

    R = np.array([[0.05]], dtype=np.float32)

    for box in results.boxes:
        if int(box.cls[0]) != 0:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        track_id = int(box.id[0]) if box.id is not None else -1
        if track_id == -1:
            continue

        foot_x = (x1 + x2) // 2
        raw_foot_y = y2

        # ── distance ───────────────────────────────────────────────────
        if raw_foot_y > yh_s:
            z_raw = fh_s / (raw_foot_y - yh_s)
        else:
            continue

        # ── init ───────────────────────────────────────────────────────
        if track_id not in kalman_states:
            kalman_states[track_id] = np.array([z_raw, 0.0], dtype=np.float32)
            kalman_covs[track_id]   = np.eye(2, dtype=np.float32)
            prev_distances[track_id] = z_raw
            speed_state[track_id]    = 0.0

        # ── deadband ───────────────────────────────────────────────────
        prev_z = float(kalman_states[track_id][0])
        z = prev_z if abs(z_raw - prev_z) < deadband else z_raw

        x = kalman_states[track_id]
        P = kalman_covs[track_id]

        # ── predict ────────────────────────────────────────────────────
        x = F @ x
        P = F @ P @ F.T + Q

        # ── update ─────────────────────────────────────────────────────
        z_vec = np.array([z], dtype=np.float32)
        y_res = z_vec - (H @ x)
        S     = H @ P @ H.T + R
        K     = P @ H.T @ np.linalg.inv(S)

        x = x + (K @ y_res).flatten()
        P = (np.eye(2) - K @ H) @ P

        kalman_states[track_id] = x
        kalman_covs[track_id]   = P

        distance = float(x[0])

        # ── speed (EMA от delta distance) ──────────────────────────────
        raw_speed = (distance - prev_distances[track_id]) * fps
        prev_distances[track_id] = distance

        speed_state[track_id] = (
            alpha * raw_speed +
            (1 - alpha) * speed_state[track_id]
        )

        speed = speed_state[track_id]

        if abs(speed) < 0.3:
            speed = 0.0

        # ── TTC ────────────────────────────────────────────────────────
        ttc    = None
        danger = 0

        if speed < -0.1 and distance > 0:
            ttc = distance / abs(speed)
            ttc = min(ttc, 10.0)  # ограничение

            if distance < DIST_MIN or ttc < TTC_CRITICAL:
                danger = 2
            elif ttc < TTC_WARNING:
                danger = 1

        danger_levels[track_id] = danger

        # ── сглаживание позиции ───────────────────────────────────────
        alpha_pos = 0.1

        if track_id not in smooth_foot_x:
            smooth_foot_x[track_id] = float(foot_x)
            smooth_foot_y[track_id] = float(raw_foot_y)
        else:
            smooth_foot_x[track_id] = (
                alpha_pos * foot_x + (1 - alpha_pos) * smooth_foot_x[track_id]
            )
            smooth_foot_y[track_id] = (
                alpha_pos * raw_foot_y + (1 - alpha_pos) * smooth_foot_y[track_id]
            )

        sx = int(smooth_foot_x[track_id])
        sy = int(smooth_foot_y[track_id])

        # ── trajectory ─────────────────────────────────────────────────
        if track_id not in trajectories:
            trajectories[track_id] = []

        trajectories[track_id].append((sx, sy))
        if len(trajectories[track_id]) > 60:
            trajectories[track_id].pop(0)

        pts = trajectories[track_id]
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], (0, 255, 255), 2)

        # ── draw ───────────────────────────────────────────────────────
        box_colors = {
            0: (0, 255, 0),
            1: (0, 165, 255),
            2: (0, 0, 255),
        }
        box_color = box_colors[danger]

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        dist_text  = f"{distance:.1f}m"
        speed_text = f"{speed * 3.6:+.1f}km/h"

        label = f"#{track_id}  {dist_text}  {speed_text}"

        if ttc is not None and danger > 0:
            label += f"  TTC:{ttc:.1f}s"

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            box_color,
            2
        )

        # ── глобальное предупреждение ─────────────────────────────────────
        h, w = frame.shape[:2]

        banner_h = 60
        y1 = h - banner_h
        y2 = h

        if danger_levels:
            max_danger = max(danger_levels.values())
        else:
            max_danger = 0

        if max_danger == 2:
            cv2.rectangle(frame, (0, y1), (w, y2), (0, 0, 200), -1)
            cv2.putText(
                frame,
                "! DANGER: PEDESTRIAN ON PATH!",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )

        elif max_danger == 1:
            cv2.rectangle(frame, (0, y1), (w, y2), (0, 100, 200), -1)
            cv2.putText(
                frame,
                "CAUTION: Pedestrian approaching",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

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
        self.error.emit("Не удалось открыть видеофайл.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 1:
        fps = 25.0  

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

    processed = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(
            frame,
            self.model,
            self.calib_data,
            fps
        )

        out.write(processed_frame)

        processed += 1

        if total_frames > 0:
            progress = int(processed / total_frames * 100)
            self.progress_changed.emit(progress)

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

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1:
            fps = 30.0  

        self._running = True

        while self._running:
            ret, frame = cap.read()
            if ret:
                processed = process_frame(
                    frame,
                    self.model,
                    self.calib_data,
                    fps
                )
                self.frame_ready.emit(processed)

        cap.release()

    def stop(self):
        self._running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Система детекции пешеходов")
        self.setGeometry(100, 100, 860, 640)

        self.calib_data   = None
        self.model        = None
        self.cam_thread   = None
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
        if not MODEL_PATH.exists():
            QMessageBox.critical(
                self, "Модель не найдена",
                f"Файл модели не найден:\n{MODEL_PATH}"
            )
            return False

        try:
            self.model = YOLO(str(MODEL_PATH))
        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки модели", str(e))
            return False

        if CALIBRATION_PATH.exists():
            try:
                with open(CALIBRATION_PATH, "r") as f:
                    self.calib_data = json.load(f)
            except Exception:
                self.calib_data = None

        return True

    def _calibration_ready(self) -> bool:
        if self.calib_data is None:
            QMessageBox.warning(
                self, "Требуется калибровка",
                "Данные калибровки не найдены.\n\n"
                "Сначала выполните калибровку камеры,\n"
                "затем нажмите «Обновить калибровку»."
            )
            return False
        return True

    def _build_menu(self):
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(14)

        title = QLabel("Система детекции пешеходов")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 26px; font-weight: bold; margin-bottom: 10px;")
        lay.addWidget(title)

        self.lbl_calib_status = QLabel()
        self.lbl_calib_status.setAlignment(Qt.AlignCenter)
        self.lbl_calib_status.setStyleSheet("font-size: 12px;")
        self._update_calib_status_label()
        lay.addWidget(self.lbl_calib_status)

        btn_video = QPushButton("📹  Обработка видеофайла")
        btn_video.setFixedSize(320, 60)
        btn_video.setStyleSheet(self._btn("#2980b9"))
        btn_video.clicked.connect(self._go_video)

        btn_cam = QPushButton("📷  Камера в реальном времени")
        btn_cam.setFixedSize(320, 60)
        btn_cam.setStyleSheet(self._btn("#27ae60"))
        btn_cam.clicked.connect(self._go_camera)

        btn_calib = QPushButton("🎯  Калибровка камеры")
        btn_calib.setFixedSize(320, 60)
        btn_calib.setStyleSheet(self._btn("#8e44ad"))
        btn_calib.clicked.connect(self._open_calibration)

        btn_reload = QPushButton("🔄  Обновить калибровку")
        btn_reload.setFixedSize(320, 40)
        btn_reload.setStyleSheet(self._btn("#555"))
        btn_reload.clicked.connect(self._reload_calibration)

        lay.addWidget(btn_video,  alignment=Qt.AlignCenter)
        lay.addWidget(btn_cam,    alignment=Qt.AlignCenter)
        lay.addWidget(btn_calib,  alignment=Qt.AlignCenter)
        lay.addWidget(btn_reload, alignment=Qt.AlignCenter)

        self.stack.addWidget(w)

    def _update_calib_status_label(self):
        if self.calib_data is not None:
            self.lbl_calib_status.setText("✅  Калибровка загружена")
            self.lbl_calib_status.setStyleSheet("font-size: 12px; color: #2ecc71;")
        else:
            self.lbl_calib_status.setText("⚠️  Калибровка не найдена — сначала выполните калибровку")
            self.lbl_calib_status.setStyleSheet("font-size: 12px; color: #e74c3c;")

    def _build_video_screen(self):
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(12)
        lay.setContentsMargins(20, 20, 20, 20)

        top = QHBoxLayout()
        btn_back = QPushButton("← Назад")
        btn_back.setFixedWidth(90)
        btn_back.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        top.addWidget(btn_back)
        top.addWidget(QLabel("<h2>Обработка видео</h2>"))
        top.addStretch()
        lay.addLayout(top)

        row_in = QHBoxLayout()
        self.lbl_in = QLabel("Файл не выбран")
        self.lbl_in.setStyleSheet("color: #aaa;")
        btn_in = QPushButton("Выбрать входное видео")
        btn_in.clicked.connect(self._select_input)
        row_in.addWidget(btn_in)
        row_in.addWidget(self.lbl_in, stretch=1)
        lay.addLayout(row_in)

        row_out = QHBoxLayout()
        self.lbl_out = QLabel("Путь сохранения не выбран")
        self.lbl_out.setStyleSheet("color: #aaa;")
        btn_out = QPushButton("Выбрать путь сохранения")
        btn_out.clicked.connect(self._select_output)
        row_out.addWidget(btn_out)
        row_out.addWidget(self.lbl_out, stretch=1)
        lay.addLayout(row_out)

        self.btn_process = QPushButton("▶  Начать обработку")
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
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(10)
        lay.setContentsMargins(20, 20, 20, 20)

        top = QHBoxLayout()
        btn_back = QPushButton("← Назад")
        btn_back.setFixedWidth(90)
        btn_back.clicked.connect(self._stop_cam_and_back)
        top.addWidget(btn_back)
        top.addWidget(QLabel("<h2>Детекция в реальном времени</h2>"))
        top.addStretch()
        lay.addLayout(top)

        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Камера:"))
        self.combo_cam = QComboBox()
        self._detect_cameras()
        cam_row.addWidget(self.combo_cam)
        self.btn_toggle = QPushButton("▶  Запустить")
        self.btn_toggle.setStyleSheet(self._btn("#27ae60"))
        self.btn_toggle.clicked.connect(self._toggle_camera)
        cam_row.addWidget(self.btn_toggle)
        cam_row.addStretch()
        lay.addLayout(cam_row)

        self.lbl_feed = QLabel("Здесь появится изображение с камеры")
        self.lbl_feed.setAlignment(Qt.AlignCenter)
        self.lbl_feed.setStyleSheet("background: #111; color: #555;")
        self.lbl_feed.setMinimumSize(640, 480)
        lay.addWidget(self.lbl_feed, stretch=1)

        self.stack.addWidget(w)

    def _go_video(self):
        if self._calibration_ready():
            self.stack.setCurrentIndex(1)

    def _go_camera(self):
        if self._calibration_ready():
            self.stack.setCurrentIndex(2)

    def _open_calibration(self):
        calib_script = BASE_PATH / "camera_calibration" / "camera_calibration.py"
        if not calib_script.exists():
            QMessageBox.critical(
                self, "Ошибка",
                f"Скрипт калибровки не найден:\n{calib_script}"
            )
            return
        subprocess.Popen(
            [sys.executable, str(calib_script)],
            cwd=str(calib_script.parent)
        )

    def _reload_calibration(self):
        if not CALIBRATION_PATH.exists():
            QMessageBox.warning(
                self, "Предупреждение",
                "Файл calibration.json не найден.\nСначала выполните калибровку."
            )
            return
        try:
            with open(CALIBRATION_PATH, "r") as f:
                self.calib_data = json.load(f)
            self._update_calib_status_label()
            QMessageBox.information(self, "Готово", "Калибровка успешно обновлена.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить калибровку:\n{e}")

    def _select_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выбрать видео", "", "Видео (*.mp4 *.avi *.mov)"
        )
        if path:
            self.lbl_in.setText(path)
            self.lbl_in.setStyleSheet("color: white;")

    def _select_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить как", "output.mp4", "Видео (*.mp4)"
        )
        if path:
            self.lbl_out.setText(path)
            self.lbl_out.setStyleSheet("color: white;")

    def _start_video(self):
        in_p  = self.lbl_in.text()
        out_p = self.lbl_out.text()

        if not os.path.exists(in_p):
            QMessageBox.warning(self, "Предупреждение", "Сначала выберите корректный видеофайл.")
            return
        if out_p in ("Путь сохранения не выбран", ""):
            QMessageBox.warning(self, "Предупреждение", "Сначала выберите путь сохранения.")
            return

        self.btn_process.setEnabled(False)
        self.btn_process.setText("Обработка…")
        self.progress.setValue(0)

        self.video_thread = VideoWorker(in_p, out_p, self.model, self.calib_data)
        self.video_thread.progress_changed.connect(self.progress.setValue)
        self.video_thread.finished.connect(self._on_video_done)
        self.video_thread.error.connect(self._on_video_error)
        self.video_thread.start()

    def _on_video_done(self, path: str):
        self.btn_process.setEnabled(True)
        self.btn_process.setText("▶  Начать обработку")
        self.progress.setValue(100)
        QMessageBox.information(self, "Готово", f"Видео сохранено:\n{path}")

    def _on_video_error(self, msg: str):
        self.btn_process.setEnabled(True)
        self.btn_process.setText("▶  Начать обработку")
        QMessageBox.critical(self, "Ошибка", msg)

    def _detect_cameras(self):
        self.combo_cam.clear()
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                self.combo_cam.addItem(f"Камера {i}", i)
                cap.release()
        if self.combo_cam.count() == 0:
            self.combo_cam.addItem("Камеры не найдены", -1)

    def _toggle_camera(self):
        if self.cam_thread and self.cam_thread.isRunning():
            self.cam_thread.stop()
            self.cam_thread = None
            self.btn_toggle.setText("▶  Запустить")
            self.btn_toggle.setStyleSheet(self._btn("#27ae60"))
            self.lbl_feed.setText("Камера остановлена")
        else:
            idx = self.combo_cam.currentData()
            if idx == -1:
                QMessageBox.warning(self, "Предупреждение", "Камеры недоступны.")
                return
            self.cam_thread = CameraWorker(idx, self.model, self.calib_data)
            self.cam_thread.frame_ready.connect(self._update_feed)
            self.cam_thread.start()
            self.btn_toggle.setText("⏹  Остановить")
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
            self.btn_toggle.setText("▶  Запустить")
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
            f"QPushButton:hover:enabled {{ background-color: {color}dd; }}"
        )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    pal = QPalette()
    pal.setColor(QPalette.Window,          QColor(45,  45,  45))
    pal.setColor(QPalette.WindowText,      QColor(220, 220, 220))
    pal.setColor(QPalette.Base,            QColor(30,  30,  30))
    pal.setColor(QPalette.Text,            QColor(220, 220, 220))
    pal.setColor(QPalette.Button,          QColor(60,  60,  60))
    pal.setColor(QPalette.ButtonText,      QColor(220, 220, 220))
    pal.setColor(QPalette.Highlight,       QColor(80,  80,  200))
    pal.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(pal)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())