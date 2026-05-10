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
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QComboBox, QStackedWidget, QMessageBox, QProgressBar,
                             QDialog, QFormLayout, QDoubleSpinBox, QCheckBox, QFrame)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QUrl
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor, QFont
from PyQt5.QtMultimedia import QSoundEffect
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

BASE_PATH        = Path(__file__).resolve().parent
CALIBRATION_PATH = BASE_PATH / "camera_calibration" / "calibration.json"
MODEL_PATH      = BASE_PATH / "neural_networks" / "yolov8" / "yolov8n.pt"
SETTINGS_PATH = BASE_PATH / "settings.json"

def load_settings():
    if SETTINGS_PATH.exists():
        with open(SETTINGS_PATH, "r") as f:
            return json.load(f)
    else:
        return {
            "TTC_CRITICAL": 2.0,
            "TTC_WARNING": 6.0,
            "DIST_MIN": 1.0,
            "sound_enabled": True
        }

def save_settings(settings):
    with open(SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=4)


# ── Глобальные состояния ──────────────────────────────────────────────────────
prev_distances  = {}
speed_state     = {}
kalman_states   = {}
kalman_covs     = {}
smooth_foot_x   = {}
smooth_foot_y   = {}
danger_levels   = {}
track_last_seen = {}   

_frame_counter  = 0

MAX_TRACK_AGE   = 90   

CAR_WIDTH_M = 0.5  # Ширина автомобиля в метрах
MAX_RADAR_DIST = 5.0 # Максимальная дистанция на радаре

_kalman_H = np.array([[1, 0]], dtype=np.float32)
_kalman_R = np.array([[0.05]], dtype=np.float32)
_kalman_I = np.eye(2, dtype=np.float32)

def get_metric_coordinates(foot_x, distance, frame_w, fh_s, calib_data):
    cam_h = calib_data.get("camera_height_m", 0.5)
    focal_px = fh_s / cam_h
    cx = frame_w / 2
    # X = (смещение_px) * Z / f
    real_x = (foot_x - cx) * distance / focal_px
    return real_x, distance

def draw_safe_corridor(frame: np.ndarray, calib_data: dict,
                       fh_s: float, yh_s: float) -> None:
    h, w    = frame.shape[:2]
    cam_h   = calib_data.get("camera_height_m", 0.5)
    focal   = fh_s / cam_h
    cx      = w / 2
    pts     = []

    for z in [15.0, 0.3]:
        x_off = (CAR_WIDTH_M / 2) * focal / z
        y_px  = int(fh_s / z + yh_s)
        pts.append([int(cx - x_off), y_px])
        pts.insert(0, [int(cx + x_off), y_px])

    pts     = np.array(pts, np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], (60, 60, 60))
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.polylines(frame, [pts], True, (160, 160, 160), 1)

def create_radar_frame(pedestrians: dict) -> np.ndarray:
    """
    pedestrians: {track_id: (real_x_m, dist_m, danger)}
    Возвращает BGR изображение радара.
    """
    RW, RH    = 320, 420
    radar     = np.full((RH, RW, 3), 18, dtype=np.uint8)
    cx        = RW // 2
    cy        = RH - 25
    m_to_py   = (RH - 40) / MAX_RADAR_DIST   # пикселей на метр по Y
    m_to_px   = RW / 8.0                      # пикселей на метр по X (8м — ширина)

    # Сетка расстояний
    for d in range(1, int(MAX_RADAR_DIST) + 1):
        y_line = cy - int(d * m_to_py)
        if y_line < 5:
            continue
        col = (55, 55, 55) if d % 2 != 0 else (85, 85, 85)
        cv2.line(radar, (20, y_line), (RW - 20, y_line), col, 1)
        if d % 2 == 0:
            cv2.putText(radar, f"{d}m", (4, y_line + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

    # Коридор на радаре
    car_half = int((CAR_WIDTH_M / 2) * m_to_px)
    cv2.rectangle(radar, (cx - car_half, 10), (cx + car_half, cy), (0, 38, 0), -1)
    cv2.line(radar, (cx, 10), (cx, cy), (50, 50, 50), 1)

    # Значок автомобиля
    cv2.rectangle(radar, (cx - 8, cy - 12), (cx + 8, cy), (160, 160, 160), -1)
    cv2.putText(radar, "CAR", (cx - 10, cy + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

    # Пешеходы
    for tid, (rx_m, rz_m, danger) in pedestrians.items():
        if rz_m <= 0 or rz_m > MAX_RADAR_DIST:
            continue
        px = cx + int(rx_m * m_to_px)
        py = cy - int(rz_m * m_to_py)
        px = int(np.clip(px, 5, RW - 5))
        py = int(np.clip(py, 5, RH - 5))
        color = {0: (0, 220, 0), 1: (0, 165, 255), 2: (0, 0, 255)}[danger]
        cv2.circle(radar, (px, py), 6, color, -1)
        cv2.putText(radar, f"#{tid}", (px + 8, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (230, 230, 230), 1)

    # Подпись
    cv2.putText(radar, "TOP VIEW", (cx - 24, RH - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 100), 1)
    cv2.rectangle(radar, (0, 0), (RW - 1, RH - 1), (70, 70, 70), 1)

    return radar

def _cleanup_stale_tracks(active_ids: set) -> None:
    global _frame_counter
    _frame_counter += 1

    for tid in active_ids:
        track_last_seen[tid] = _frame_counter

    stale = [
        tid for tid, last in track_last_seen.items()
        if _frame_counter - last > MAX_TRACK_AGE
    ]
    for tid in stale:
        for d in (kalman_states, kalman_covs, prev_distances, speed_state,
                  smooth_foot_x, smooth_foot_y, track_last_seen):
            d.pop(tid, None)

def put_russian_text(frame, text, pos, color=(255,255,255)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font = ImageFont.truetype("font.ttf", 24)

    draw.text(pos, text, font=font, fill=color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def process_frame(frame: np.ndarray, model, calib_data: dict,
                  fps: float, settings: dict):
    """
    Возвращает (processed_frame, radar_frame, max_danger).
    """
    TTC_CRITICAL = settings["TTC_CRITICAL"]
    TTC_WARNING  = settings["TTC_WARNING"]
    DIST_MIN     = settings["DIST_MIN"]

    fh        = calib_data["fh"]
    y_horizon = calib_data["y_horizon"]
    frame_h, frame_w = frame.shape[:2]
    calib_h   = calib_data.get("image_height", frame_h)
    scale     = frame_h / calib_h if calib_h > 0 else 1.0
    fh_s      = fh * scale
    yh_s      = y_horizon * scale
    cam_h     = calib_data.get("camera_height_m", 0.5)
    focal_px  = fh_s / cam_h

    # Коридор безопасности
    draw_safe_corridor(frame, calib_data, fh_s, yh_s)

    results = model.track(
        frame, conf=0.4, verbose=False, device=0,
        tracker="bytetrack.yaml", persist=True, amp=False
    )[0]

    danger_levels.clear()
    active_pedestrians: dict = {}   # {track_id: (real_x, dist, danger)}

    if results.boxes is None or len(results.boxes) == 0:
        radar = create_radar_frame({})
        return frame, radar, 0

    # Активные треки
    active_ids = {
        int(b.id[0]) for b in results.boxes
        if b.id is not None and int(b.cls[0]) == 0
    }
    _cleanup_stale_tracks(active_ids)

    dt      = 1.0 / max(fps, 1e-5)
    alpha   = 0.2
    sigma_a = 0.5
    alpha_p = 0.1

    F = np.array([[1, dt], [0, 1]], dtype=np.float32)
    Q = np.array([
        [dt**4 / 4, dt**3 / 2],
        [dt**3 / 2, dt**2]
    ], dtype=np.float32) * sigma_a**2

    for box in results.boxes:
        if int(box.cls[0]) != 0:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        track_id = int(box.id[0]) if box.id is not None else -1
        if track_id == -1:
            continue

        foot_x     = (x1 + x2) // 2
        raw_foot_y = y2

        # Критически близкий пешеход
        if raw_foot_y >= frame_h * 0.99:
            real_x = (foot_x - frame_w / 2) * 0.5 / focal_px
            danger = 2 if abs(real_x) <= CAR_WIDTH_M / 2 else 1
            danger_levels[track_id] = danger
            active_pedestrians[track_id] = (real_x, 0.5, danger)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"#{track_id}  <1m",
                        (x1, max(y1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(frame, f"#{track_id}  <1m",
                        (x1, max(y1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
            continue

        if raw_foot_y <= yh_s:
            continue

        z_raw = fh_s / (raw_foot_y - yh_s)

        # Инициализация Kalman
        if track_id not in kalman_states:
            kalman_states[track_id]  = np.array([z_raw, 0.0], dtype=np.float32)
            kalman_covs[track_id]    = np.eye(2, dtype=np.float32)
            prev_distances[track_id] = z_raw
            speed_state[track_id]    = 0.0
            smooth_foot_x[track_id] = float(foot_x)
            smooth_foot_y[track_id] = float(raw_foot_y)

        # Deadband
        prev_z = float(kalman_states[track_id][0])
        z = prev_z if abs(z_raw - prev_z) < 0.03 else z_raw

        # Kalman predict + update
        x_k = F @ kalman_states[track_id]
        P_k = F @ kalman_covs[track_id] @ F.T + Q
        inn = np.array([z], dtype=np.float32) - (_kalman_H @ x_k)
        S   = _kalman_H @ P_k @ _kalman_H.T + _kalman_R
        K   = P_k @ _kalman_H.T / float(S[0, 0])
        kalman_states[track_id] = x_k + K.flatten() * float(inn[0])
        kalman_covs[track_id]   = (_kalman_I - np.outer(K, _kalman_H)) @ P_k

        distance = float(kalman_states[track_id][0])

        # Скорость EMA
        raw_speed = (distance - prev_distances[track_id]) * fps
        prev_distances[track_id] = distance
        speed_state[track_id]    = alpha * raw_speed + (1 - alpha) * speed_state[track_id]
        speed = speed_state[track_id] if abs(speed_state[track_id]) >= 0.3 else 0.0

        # TTC и опасность
        ttc    = None
        danger = 0
        if speed < -0.1 and distance > 0:
            ttc = min(distance / abs(speed), 10.0)
            if distance < DIST_MIN or ttc < TTC_CRITICAL:
                danger = 2
            elif ttc < TTC_WARNING:
                danger = 1

        # Коридор безопасности
        real_x = (foot_x - frame_w / 2) * distance / focal_px
        if abs(real_x) > CAR_WIDTH_M / 2:
            danger = min(danger, 1)

        danger_levels[track_id] = danger
        active_pedestrians[track_id] = (real_x, distance, danger)

        # Сглаживание позиции
        smooth_foot_x[track_id] = alpha_p * foot_x     + (1 - alpha_p) * smooth_foot_x[track_id]
        smooth_foot_y[track_id] = alpha_p * raw_foot_y + (1 - alpha_p) * smooth_foot_y[track_id]

        # Отрисовка бокса
        box_color = {0: (0, 255, 0), 1: (0, 165, 255), 2: (0, 0, 255)}[danger]
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        label = f"#{track_id}  {distance:.1f}m  {speed * 3.6:+.1f}km/h"
        if ttc is not None and danger > 0:
            label += f"  TTC:{ttc:.1f}s"

        cv2.putText(frame, label, (x1, max(y1 - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3)
        cv2.putText(frame, label, (x1, max(y1 - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, box_color, 1)

    radar      = create_radar_frame(active_pedestrians)
    max_danger = max(danger_levels.values(), default=0)

    return frame, radar, max_danger

class SettingsDialog(QDialog):
    def __init__(self, settings):
        super().__init__()
        self.setWindowTitle("Настройки")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self.settings = settings

        layout = QFormLayout()

        self.ttc_critical = QDoubleSpinBox()
        self.ttc_critical.setValue(settings["TTC_CRITICAL"])

        self.ttc_warning = QDoubleSpinBox()
        self.ttc_warning.setValue(settings["TTC_WARNING"])

        self.dist_min = QDoubleSpinBox()
        self.dist_min.setValue(settings["DIST_MIN"])

        self.sound_checkbox = QCheckBox("Включить звук")
        self.sound_checkbox.setChecked(settings["sound_enabled"])

        layout.addRow("TTC критический:", self.ttc_critical)
        layout.addRow("TTC предупреждение:", self.ttc_warning)
        layout.addRow("Мин. дистанция:", self.dist_min)
        layout.addRow(self.sound_checkbox)

        btn_save = QPushButton("Сохранить")
        btn_save.setFixedSize(120, 60)
        btn_save.setStyleSheet(self._btn("#555"))
        btn_save.clicked.connect(self.save)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(btn_save)
        btn_layout.addStretch()
        
        layout.addRow(btn_layout)
        self.setLayout(layout)

    def save(self):
        self.settings["TTC_CRITICAL"] = self.ttc_critical.value()
        self.settings["TTC_WARNING"]  = self.ttc_warning.value()
        self.settings["DIST_MIN"]     = self.dist_min.value()
        self.settings["sound_enabled"] = self.sound_checkbox.isChecked()

        save_settings(self.settings)
        self.accept()
    
    @staticmethod
    def _btn(color: str) -> str:
        return (
            f"QPushButton {{"
            f"  background-color: {color}; color: white;"
            f"  border-radius: 6px; font-size: 14px;"
            f"  font-weight: bold; padding: 6px 14px;"
            f"}}"
            f"QPushButton:disabled {{ background-color: #444; color: #888; }}"
            f"QPushButton:hover:enabled {{ background-color: {color}; opacity: 0.9;}}"
        )

class VideoWorker(QThread):
    frame_ready      = pyqtSignal(object)
    progress_changed = pyqtSignal(int)
    finished         = pyqtSignal(str)
    error            = pyqtSignal(str)

    def __init__(self, input_path: str, output_path: str,
                 model, calib_data: dict, settings: dict):
        super().__init__()
        self.input_path  = input_path
        self.output_path = output_path
        self.model       = model
        self.calib_data  = calib_data
        self.settings    = settings

    def run(self):
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            self.error.emit("Не удалось открыть видеофайл.")
            return

        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out    = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        processed = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            f, radar, danger = process_frame(frame, self.model,
                                             self.calib_data, fps, self.settings)
            out.write(f)
            self.frame_ready.emit((f, radar, danger))
            processed += 1
            if total > 0:
                self.progress_changed.emit(int(processed / total * 100))

        cap.release()
        out.release()
        self.finished.emit(self.output_path)


class CameraWorker(QThread):
    frame_ready = pyqtSignal(object)

    def __init__(self, camera_index: int, model, calib_data: dict, settings: dict):
        super().__init__()
        self.camera_index = camera_index
        self.model        = model
        self.calib_data   = calib_data
        self.settings     = settings
        self._running     = False

    def run(self):
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._running = True
        while self._running:
            ret, frame = cap.read()
            if ret:
                data = process_frame(frame, self.model,
                                     self.calib_data, fps, self.settings)
                self.frame_ready.emit(data)
        cap.release()

    def stop(self):
        self._running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Система детекции пешеходов")
        self.setMinimumSize(1280, 760)

        self.settings     = load_settings()
        self.calib_data   = None
        self.model        = None
        self.cam_thread   = None
        self.video_thread = None
        self.last_sound_t = 0

        self._init_sounds()
        if not self._load_resources():
            sys.exit()

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self._build_menu()       # index 0
        self._build_dashboard()  # index 1
        self.stack.setCurrentIndex(0)

    # ── Инициализация ─────────────────────────────────────────────────────────

    def _init_sounds(self):
        self.snd_warning = QSoundEffect()
        self.snd_warning.setSource(QUrl.fromLocalFile("warning.wav"))
        self.snd_warning.setVolume(0.5)
        self.snd_danger  = QSoundEffect()
        self.snd_danger.setSource(QUrl.fromLocalFile("danger.wav"))
        self.snd_danger.setVolume(0.8)

    def _load_resources(self) -> bool:
        if not MODEL_PATH.exists():
            QMessageBox.critical(self, "Ошибка", f"Модель не найдена:\n{MODEL_PATH}")
            return False
        try:
            self.model = YOLO(str(MODEL_PATH))
        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки модели", str(e))
            return False
        if CALIBRATION_PATH.exists():
            try:
                with open(CALIBRATION_PATH) as f:
                    self.calib_data = json.load(f)
            except Exception:
                self.calib_data = None
        return True

    def _calibration_ready(self) -> bool:
        if self.calib_data is None:
            QMessageBox.warning(self, "Требуется калибровка",
                                "Сначала выполните калибровку камеры.")
            return False
        return True

    # ── Экран меню ────────────────────────────────────────────────────────────

    def _build_menu(self):
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(16)

        title = QLabel("Система детекции пешеходов")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size:28px; font-weight:bold; margin-bottom:10px;")
        lay.addWidget(title)

        # Статус калибровки
        self.lbl_calib = QLabel()
        self.lbl_calib.setAlignment(Qt.AlignCenter)
        self._refresh_calib_label()
        lay.addWidget(self.lbl_calib)

        for text, color, slot in [
            ("📹  Обработка видеофайла",       "#2980b9", self._go_video),
            ("📷  Камера в реальном времени",   "#27ae60", self._go_camera),
            ("🎯  Калибровка камеры",            "#8e44ad", self._open_calibration),
            ("🔄  Обновить калибровку",          "#555555", self._reload_calibration),
            ("⚙️  Настройки",                   "#283f67", self._open_settings),
        ]:
            btn = QPushButton(text)
            btn.setFixedSize(340, 58)
            btn.setStyleSheet(self._btn(color))
            btn.clicked.connect(slot)
            lay.addWidget(btn, alignment=Qt.AlignCenter)

        self.stack.addWidget(w)

    def _refresh_calib_label(self):
        if self.calib_data:
            self.lbl_calib.setText("✅  Калибровка загружена")
            self.lbl_calib.setStyleSheet("font-size:12px; color:#2ecc71;")
        else:
            self.lbl_calib.setText("⚠️  Калибровка не найдена")
            self.lbl_calib.setStyleSheet("font-size:12px; color:#e74c3c;")

    # ── Dashboard ─────────────────────────────────────────────────────────────

    def _build_dashboard(self):
        """
        Layout:
        ┌─────────────────────┬──────────────┐
        │                     │  Зона 2      │
        │   Зона 1 (видео)    │  Радар       │
        │                     ├──────────────┤
        │                     │  Зона 3      │
        │                     │  Инфо-панель │
        └─────────────────────┴──────────────┘
        """
        page = QWidget()
        root = QHBoxLayout(page)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # ── Зона 1: Видео ─────────────────────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(8)

        self.lbl_video = QLabel()
        self.lbl_video.setMinimumSize(860, 640)
        self.lbl_video.setStyleSheet(
            "background:#000; border:2px solid #333; border-radius:6px;"
        )
        self.lbl_video.setAlignment(Qt.AlignCenter)
        left.addWidget(self.lbl_video)

        # Прогресс-бар для видео
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.setStyleSheet(
            "QProgressBar { border:none; background:#222; border-radius:3px; }"
            "QProgressBar::chunk { background:#27ae60; border-radius:3px; }"
        )
        left.addWidget(self.progress)

        root.addLayout(left, stretch=3)

        # ── Правая колонка ────────────────────────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(12)

        # Зона 2: Радар
        self.lbl_radar = QLabel()
        self.lbl_radar.setFixedSize(320, 420)
        self.lbl_radar.setStyleSheet(
            "background:#111; border:1px solid #444; border-radius:6px;"
        )
        self.lbl_radar.setAlignment(Qt.AlignCenter)
        right.addWidget(self.lbl_radar)

        # Зона 3: Информационная панель
        self.warn_panel = QFrame()
        self.warn_panel.setFixedSize(320, 160)
        
        warn_lay = QVBoxLayout(self.warn_panel)
        warn_lay.setContentsMargins(10, 10, 10, 10)

        self.lbl_warn = QLabel("ПУТЬ СВОБОДЕН")
        self.lbl_warn.setAlignment(Qt.AlignCenter)
        self.lbl_warn.setWordWrap(True)
        self.lbl_warn.setFont(QFont("Segoe UI", 13, QFont.Bold))
        warn_lay.addWidget(self.lbl_warn)

        # !!! ВАЖНО: Вызываем настройку стиля только ПОСЛЕ создания lbl_warn !!!
        self._set_warn_style(0)

        right.addWidget(self.warn_panel)

        # Выбор камеры
        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Камера:"))
        self.combo_cam = QComboBox()
        self._detect_cameras()
        cam_row.addWidget(self.combo_cam, stretch=1)
        right.addLayout(cam_row)

        # Кнопки управления
        self.btn_toggle = QPushButton("▶  Запустить камеру")
        self.btn_toggle.setStyleSheet(self._btn("#27ae60"))
        self.btn_toggle.clicked.connect(self._toggle_camera)
        right.addWidget(self.btn_toggle)

        btn_back = QPushButton("← Назад в меню")
        btn_back.setStyleSheet(self._btn("#555"))
        btn_back.clicked.connect(self._stop_and_back)
        right.addWidget(btn_back)

        right.addStretch()
        root.addLayout(right, stretch=1)

        self.stack.addWidget(page)

    def _set_warn_style(self, danger: int):
        styles = {
            0: ("background:#0a2a0a; border:2px solid #2ecc71; border-radius:10px;",
                "color:#00e676; border:none;", "ПУТЬ СВОБОДЕН"),
            1: ("background:#2a1a00; border:2px solid #f39c12; border-radius:10px;",
                "color:#f39c12; border:none;", "ВНИМАНИЕ:\nПЕШЕХОД НА ПУТИ"),
            2: ("background:#2a0000; border:3px solid #e74c3c; border-radius:10px;",
                "color:#ff4444; border:none;", "ОПАСНОСТЬ:\nЭКСТРЕННОЕ ТОРМОЖЕНИЕ"),
        }
        panel_style, text_style, text = styles[danger]
        self.warn_panel.setStyleSheet(panel_style)
        self.lbl_warn.setStyleSheet(text_style)
        self.lbl_warn.setText(text)

    # ── Слоты меню ────────────────────────────────────────────────────────────

    def _go_video(self):
        if not self._calibration_ready():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Выбрать видео", "", "Видео (*.mp4 *.avi *.mov)"
        )
        if not path:
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результат", "output.mp4", "Видео (*.mp4)"
        )
        if not out_path:
            return
        self.stack.setCurrentIndex(1)
        self.progress.setVisible(True)
        self.progress.setValue(0)

        self.video_thread = VideoWorker(path, out_path, self.model,
                                        self.calib_data, self.settings.copy())
        self.video_thread.frame_ready.connect(self._update_dashboard)
        self.video_thread.progress_changed.connect(self.progress.setValue)
        self.video_thread.finished.connect(self._on_video_done)
        self.video_thread.error.connect(self._on_video_error)
        self.video_thread.start()

    def _go_camera(self):
        if not self._calibration_ready():
            return
        self.stack.setCurrentIndex(1)

    def _open_calibration(self):
        script = BASE_PATH / "camera_calibration" / "camera_calibration.py"
        if not script.exists():
            QMessageBox.critical(self, "Ошибка", f"Скрипт не найден:\n{script}")
            return
        subprocess.Popen([sys.executable, str(script)], cwd=str(script.parent))

    def _reload_calibration(self):
        if not CALIBRATION_PATH.exists():
            QMessageBox.warning(self, "Предупреждение",
                                "calibration.json не найден.")
            return
        try:
            with open(CALIBRATION_PATH) as f:
                self.calib_data = json.load(f)
            self._refresh_calib_label()
            QMessageBox.information(self, "Готово", "Калибровка обновлена.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    def _open_settings(self):
        dlg = SettingsDialog(self.settings, self)
        if dlg.exec_():
            self.settings = load_settings()

    # ── Слоты dashboard ───────────────────────────────────────────────────────

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
            self.btn_toggle.setText("▶  Запустить камеру")
            self.btn_toggle.setStyleSheet(self._btn("#27ae60"))
            self._set_warn_style(0)
        else:
            if not self._calibration_ready():
                return
            idx = self.combo_cam.currentData()
            if idx == -1:
                QMessageBox.warning(self, "Предупреждение", "Камеры не найдены.")
                return
            self.cam_thread = CameraWorker(idx, self.model,
                                           self.calib_data, self.settings.copy())
            self.cam_thread.frame_ready.connect(self._update_dashboard)
            self.cam_thread.start()
            self.btn_toggle.setText("⏹  Остановить")
            self.btn_toggle.setStyleSheet(self._btn("#c0392b"))

    def _update_dashboard(self, data):
        frame, radar, danger = data

        # Зона 1 — видео
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.lbl_video.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.lbl_video.width(), self.lbl_video.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

        # Зона 2 — радар
        rgb_r = cv2.cvtColor(radar, cv2.COLOR_BGR2RGB)
        qr    = QImage(rgb_r.data, rgb_r.shape[1], rgb_r.shape[0],
                       rgb_r.shape[1] * 3, QImage.Format_RGB888)
        self.lbl_radar.setPixmap(QPixmap.fromImage(qr))

        # Зона 3 — информационная панель
        self._set_warn_style(danger)

        # Звук
        self._play_sound(danger)

    def _play_sound(self, danger: int):
        if not self.settings.get("sound_enabled", True):
            return
        now = time.time()
        if danger == 2 and now - self.last_sound_t > 0.2:
            if self.snd_danger.isLoaded():
                self.snd_danger.play()
            self.last_sound_t = now
        elif danger == 1 and now - self.last_sound_t > 0.6:
            if self.snd_warning.isLoaded():
                self.snd_warning.play()
            self.last_sound_t = now

    def _on_video_done(self, path: str):
        self.progress.setVisible(False)
        QMessageBox.information(self, "Готово", f"Видео сохранено:\n{path}")

    def _on_video_error(self, msg: str):
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Ошибка", msg)

    def _stop_and_back(self):
        if self.cam_thread and self.cam_thread.isRunning():
            self.cam_thread.stop()
            self.cam_thread = None
            self.btn_toggle.setText("▶  Запустить камеру")
            self.btn_toggle.setStyleSheet(self._btn("#27ae60"))
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.terminate()
        self.progress.setVisible(False)
        self._set_warn_style(0)
        self.stack.setCurrentIndex(0)

    def closeEvent(self, event):
        if self.cam_thread and self.cam_thread.isRunning():
            self.cam_thread.stop()
        event.accept()

    # ── Стили ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _btn(color: str) -> str:
        return (
            f"QPushButton {{"
            f"  background-color:{color}; color:white;"
            f"  border-radius:6px; font-size:13px;"
            f"  font-weight:bold; padding:7px 14px;"
            f"}}"
            f"QPushButton:disabled {{ background:#444; color:#888; }}"
            f"QPushButton:hover:enabled {{ background:{color}cc; }}"
        )


# ── Точка входа ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    pal = QPalette()
    pal.setColor(QPalette.Window,          QColor(40,  40,  40))
    pal.setColor(QPalette.WindowText,      QColor(220, 220, 220))
    pal.setColor(QPalette.Base,            QColor(28,  28,  28))
    pal.setColor(QPalette.Text,            QColor(220, 220, 220))
    pal.setColor(QPalette.Button,          QColor(55,  55,  55))
    pal.setColor(QPalette.ButtonText,      QColor(220, 220, 220))
    pal.setColor(QPalette.Highlight,       QColor(70,  70,  190))
    pal.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(pal)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())