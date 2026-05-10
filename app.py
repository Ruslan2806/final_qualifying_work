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
                             QComboBox, QStackedWidget, QMessageBox, QProgressBar, 
                             QDialog, QFormLayout, QDoubleSpinBox, QCheckBox)

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor
from ultralytics import YOLO

from PyQt5.QtMultimedia import QSoundEffect
from PyQt5.QtCore import QUrl
import time

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

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

def draw_safe_corridor(frame: np.ndarray, calib_data: dict, fh_s: float, yh_s: float):
    h, w = frame.shape[:2]
    cam_h = calib_data.get("camera_height_m", 0.5)
    focal_px = fh_s / cam_h
    cx = w / 2

    # Формируем трапецию коридора от 0.1 до 15 метров
    pts = []
    # 15м - дальняя граница, 0.1м - ближняя 
    for z in [15.0, 0.1]: 
        x_offset = (CAR_WIDTH_M / 2) * focal_px / z
        y_px = (fh_s / z) + yh_s
        pts.append([cx - x_offset, y_px]) 
        pts.insert(0, [cx + x_offset, y_px]) 

    pts = np.array(pts, np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], (80, 80, 80)) 
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.polylines(frame, [pts], True, (180, 180, 180), 1)

def draw_radar(frame: np.ndarray, pedestrians_metrics: dict, frame_w: int):
    RW, RH = 180, 240 
    MARGIN = 15
    h, w = frame.shape[:2]
    rx, ry = w - RW - MARGIN, MARGIN

    # Фон радара
    cv2.rectangle(frame, (rx, ry), (rx + RW, ry + RH), (20, 20, 20), -1)
    cv2.rectangle(frame, (rx, ry), (rx + RW, ry + RH), (80, 80, 80), 1)

    cx_r = rx + RW // 2
    cy_r = ry + RH - 15 

    # Сетка с шагом 1 метр
    for d in range(1, int(MAX_RADAR_DIST) + 1):
        y_m = cy_r - int((d / MAX_RADAR_DIST) * (RH - 40))
        color = (60, 60, 60) if d % 2 != 0 else (90, 90, 90)
        cv2.line(frame, (rx + 10, y_m), (rx + RW - 10, y_m), color, 1)
        if d % 2 == 0:
            cv2.putText(frame, f"{d}m", (rx + 2, y_m + 3), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)

    # Коридор автомобиля на радаре
    meter_to_px = RW / 6.0 
    car_half_w = int((CAR_WIDTH_M / 2) * meter_to_px)
    cv2.rectangle(frame, (cx_r - car_half_w, ry + 5), (cx_r + car_half_w, cy_r), (0, 40, 0), -1)

    # Пешеходы
    for tid, (rx_m, rz_m, danger) in pedestrians_metrics.items():
        if rz_m > MAX_RADAR_DIST or rz_m <= 0: continue
        px = cx_r + int(rx_m * meter_to_px)
        py = cy_r - int((rz_m / MAX_RADAR_DIST) * (RH - 40))
        px = np.clip(px, rx + 5, rx + RW - 5)
        py = np.clip(py, ry + 5, ry + RH - 5)
        color = {0: (0, 255, 0), 1: (0, 165, 255), 2: (0, 0, 255)}[danger]
        cv2.circle(frame, (px, py), 4, color, -1)
        cv2.putText(frame, str(tid), (px + 5, py + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

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

def process_frame(frame: np.ndarray, model, calib_data: dict, fps: float, settings):
    TTC_CRITICAL = settings["TTC_CRITICAL"]
    TTC_WARNING  = settings["TTC_WARNING"]
    DIST_MIN     = settings["DIST_MIN"]

    fh, y_horizon = calib_data["fh"], calib_data["y_horizon"]
    frame_h, frame_w = frame.shape[:2]
    calib_h = calib_data.get("image_height", frame_h)
    scale   = frame_h / calib_h if calib_h > 0 else 1.0
    fh_s, yh_s = fh * scale, y_horizon * scale

    results = model.track(frame, conf=0.4, verbose=False, device=0, tracker="bytetrack.yaml", persist=True, amp=False)[0]

    danger_levels.clear()
    active_pedestrians_metric = {} 

    draw_safe_corridor(frame, calib_data, fh_s, yh_s)

    if results.boxes is None or len(results.boxes) == 0:
        draw_radar(frame, {}, frame_w)
        return frame, 0

    active_ids = {int(box.id[0]) for box in results.boxes if box.id is not None and int(box.cls[0]) == 0}
    _cleanup_stale_tracks(active_ids)

    dt, alpha, sigma_a = 1.0 / max(fps, 1e-5), 0.2, 0.5
    F = np.array([[1, dt], [0, 1]], dtype=np.float32)
    Q = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]], dtype=np.float32) * sigma_a**2

    for box in results.boxes:
        if int(box.cls[0]) != 0: continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        track_id = int(box.id[0]) if box.id is not None else -1
        if track_id == -1: continue

        foot_x, raw_foot_y = (x1 + x2) // 2, y2

        # Логика критической близости
        if raw_foot_y >= frame_h * 0.99:
            temp_x, _ = get_metric_coordinates(foot_x, 0.5, frame_w, fh_s, calib_data)
            danger = 2 if abs(temp_x) <= (CAR_WIDTH_M / 2) else 1
            danger_levels[track_id] = danger
            active_pedestrians_metric[track_id] = (temp_x, 0.5, danger)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            continue

        if raw_foot_y <= yh_s: continue
        z_raw = fh_s / (raw_foot_y - yh_s)

        # Фильтр Калмана
        if track_id not in kalman_states:
            kalman_states[track_id] = np.array([z_raw, 0.0], dtype=np.float32)
            kalman_covs[track_id] = np.eye(2, dtype=np.float32)
            prev_distances[track_id], speed_state[track_id] = z_raw, 0.0

        x_k = F @ kalman_states[track_id]
        P_k = F @ kalman_covs[track_id] @ F.T + Q
        inn = np.array([z_raw], dtype=np.float32) - (_kalman_H @ x_k)
        S = _kalman_H @ P_k @ _kalman_H.T + _kalman_R
        K = P_k @ _kalman_H.T / float(S[0, 0])
        kalman_states[track_id] = x_k + K.flatten() * float(inn[0])
        kalman_covs[track_id] = (_kalman_I - np.outer(K, _kalman_H)) @ P_k

        distance = float(kalman_states[track_id][0])

        # Скорость и TTC
        raw_speed = (distance - prev_distances[track_id]) * fps
        prev_distances[track_id] = distance
        speed_state[track_id] = alpha * raw_speed + (1 - alpha) * speed_state[track_id]
        speed = speed_state[track_id]

        ttc, danger = None, 0
        if speed < -0.1:
            ttc = distance / abs(speed)
            if distance < DIST_MIN or ttc < TTC_CRITICAL: danger = 2
            elif ttc < TTC_WARNING: danger = 1

        # Проверка коридора (метрическая)
        real_x, real_z = get_metric_coordinates(foot_x, distance, frame_w, fh_s, calib_data)
        if abs(real_x) > (CAR_WIDTH_M / 2):
            danger = min(danger, 1)

        danger_levels[track_id] = danger
        active_pedestrians_metric[track_id] = (real_x, real_z, danger)

        # Отрисовка
        box_color = {0: (0, 255, 0), 1: (0, 165, 255), 2: (0, 0, 255)}[danger]
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        label = f"#{track_id} {distance:.1f}m {speed*3.6:+.1f}kmh"
        if ttc and danger > 0: label += f" TTC:{ttc:.1f}s"
        cv2.putText(frame, label, (x1, max(y1-10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
        cv2.putText(frame, label, (x1, max(y1-10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

    draw_radar(frame, active_pedestrians_metric, frame_w)

    max_danger = max(danger_levels.values(), default=0)
    if max_danger == 2:
        cv2.rectangle(frame, (0, frame_h-40), (frame_w, frame_h), (0,0,200), -1)
        frame = put_russian_text(frame, "ОПАСНОСТЬ: ТОРМОЖЕНИЕ!", (10, frame_h-35))
    elif max_danger == 1:
        cv2.rectangle(frame, (0, frame_h-40), (frame_w, frame_h), (0,100,200), -1)
        frame = put_russian_text(frame, "ВНИМАНИЕ: Пешеход у пути", (10, frame_h-35))

    return frame, max_danger

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
    progress_changed = pyqtSignal(int)
    finished         = pyqtSignal(str)
    error            = pyqtSignal(str)

    def __init__(self, input_path: str, output_path: str, model, calib_data: dict, settings):
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
    
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps          = cap.get(cv2.CAP_PROP_FPS)
    
        if not fps or fps <= 1:
            fps = 25.0  
    
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out    = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
    
        processed = 0
    
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame, _ = process_frame(
                frame,
                self.model,
                self.calib_data,
                fps,
                self.settings
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
    #frame_ready = pyqtSignal(np.ndarray)
    frame_ready = pyqtSignal(object)

    def __init__(self, camera_index: int, model, calib_data: dict, settings):
        super().__init__()
        self.camera_index = camera_index
        self.model        = model
        self.calib_data   = calib_data
        self.settings = settings
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
                processed, danger  = process_frame(
                    frame,
                    self.model,
                    self.calib_data,
                    fps,
                    self.settings
                )
                self.frame_ready.emit((processed, danger))

        cap.release()

    def stop(self):
        self._running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Система детекции пешеходов")
        self.setGeometry(100, 100, 860, 640)
        self.settings = load_settings()
        self.calib_data   = None
        self.model        = None
        self.cam_thread   = None
        self.video_thread = None

        self.current_danger_level = 0
        self.sound_interval_warning = 0.5 
        self.sound_interval_danger = 0.1
        self.last_sound_time = 0

        self.sound_warning = QSoundEffect()
        self.sound_warning.setSource(QUrl.fromLocalFile("warning.wav"))
        self.sound_warning.setVolume(0.5)

        self.sound_danger = QSoundEffect()
        self.sound_danger.setSource(QUrl.fromLocalFile("danger.wav"))
        self.sound_danger.setVolume(0.7)

        self.last_danger_time = 0
        self.last_warning_time = 0

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

        btn_settings = QPushButton("⚙️ Настройки")
        btn_settings.setFixedSize(320, 60)
        btn_settings.setStyleSheet(self._btn("#283f67"))
        btn_settings.clicked.connect(self._open_settings)

        lay.addWidget(btn_video,  alignment=Qt.AlignCenter)
        lay.addWidget(btn_cam,    alignment=Qt.AlignCenter)
        lay.addWidget(btn_calib,  alignment=Qt.AlignCenter)
        lay.addWidget(btn_settings, alignment=Qt.AlignCenter)

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

    def _open_settings(self):
        dialog = SettingsDialog(self.settings)
        if dialog.exec_():
            self.settings = load_settings()  

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

        self.video_thread = VideoWorker(in_p, out_p, self.model, self.calib_data, self.settings.copy())
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
            cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
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
            self.cam_thread = CameraWorker(idx, self.model, self.calib_data, self.settings.copy())
            self.cam_thread.frame_ready.connect(self._update_feed)
            self.cam_thread.start()
            self.btn_toggle.setText("⏹  Остановить")
            self.btn_toggle.setStyleSheet(self._btn("#c0392b"))

    def _update_feed(self, data): 
        frame, danger = data

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)

        self.lbl_feed.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.lbl_feed.width(), self.lbl_feed.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

        self.current_danger_level = max(self.current_danger_level, danger)
        if self.settings["sound_enabled"]:
            now = time.time()
    
            if self.current_danger_level == 2:
                if now - self.last_sound_time > self.sound_interval_danger:
                    if self.sound_danger.isLoaded():
                        self.sound_danger.play()
                    self.last_sound_time = now
    
            elif self.current_danger_level == 1:
                if now - self.last_sound_time > self.sound_interval_warning:
                    if self.sound_warning.isLoaded():
                        self.sound_warning.play()
                    self.last_sound_time = now
        
        if danger == 0:
            self.current_danger_level = 0

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
            f"QPushButton:hover:enabled {{ background-color: {color}; opacity: 0.9;}}"
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