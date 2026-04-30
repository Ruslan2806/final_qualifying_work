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
import yaml
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QComboBox, QStackedWidget, QMessageBox, QProgressBar)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO

BASE_PATH = Path(__file__).resolve().parent
CALIBRATION_PATH = BASE_PATH / "camera_calibration" / "calibration.json"
MODEL_PATH = BASE_PATH / "neural_networks" / "yolov8" / "yolov8n.pt"


class CameraWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, camera_index, model, calib_data):
        super().__init__()
        self.camera_index = camera_index
        self.model = model
        self.calib_data = calib_data
        self.running = False

    def run(self):
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        self.running = True
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            
            processed_frame = process_frame(frame, self.model, self.calib_data)
            self.frame_ready.emit(processed_frame)
            
        cap.release()

    def stop(self):
        self.running = False
        self.wait()

class VideoWorker(QThread):
    progress_changed = pyqtSignal(int)
    finished = pyqtSignal(str)

    def __init__(self, input_path, output_path, model, calib_data):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.model = model
        self.calib_data = calib_data

    def run(self):
        cap = cv2.VideoCapture(self.input_path)
        total_frames = int(cap.get(cv2.get(cv2.CAP_PROP_FRAME_COUNT)))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        processed_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = process_frame(frame, self.model, self.calib_data)
            out.write(processed_frame)

            processed_count += 1
            progress = int((processed_count / total_frames) * 100)
            self.progress_changed.emit(progress)

        cap.release()
        out.release()
        self.finished.emit(self.output_path)

def process_frame(frame, model, calib_data):
    results = model.predict(frame, conf=0.4, verbose=False, device=0)[0]
    
    fh = calib_data['fh']
    y_horizon = calib_data['y_horizon']

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls != 0:
            continue
            
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        y_feet = y2
        
        if y_feet > y_horizon:
            distance = fh / (y_feet - y_horizon)
        else:
            distance = float('inf') 

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Вывод текста
        label_text = f"Pedestrian: {distance:.2f}m" if distance != float('inf') else "Pedestrian"
        cv2.putText(frame, label_text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    return frame

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Система детекции пешеходов (YOLOv8)")
        self.setGeometry(100, 100, 800, 600)
        
        self.calib_data = None
        self.model = None
        
        if not self.check_preconditions():
            sys.exit()

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        self.init_menu_screen()
        self.init_video_screen()
        self.init_camera_screen()
        
        self.stacked_widget.setCurrentIndex(0) 

    def check_preconditions(self):
        if not CALIBRATION_PATH.exists():
            QMessageBox.critical(self, "Ошибка", 
                f"Файл калибровки не найден по пути:\n{CALIBRATION_PATH}\n\nСначала проведите калибровку!")
            return False
            
        if not MODEL_PATH.exists():
            QMessageBox.critical(self, "Ошибка", 
                f"Модель YOLOv8 не найдена по пути:\n{MODEL_PATH}\n\nПоложите файл yolov8n.pt в папку.")
            return False

        try:
            with open(CALIBRATION_PATH, 'r') as f:
                self.calib_data = json.load(f)
            self.model = YOLO(str(MODEL_PATH))
            return True
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файлы: {e}")
            return False


    def init_menu_screen(self):
        screen = QWidget()
        layout = QVBoxLayout()
        
        title = QLabel("Выберите режим работы")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        
        btn_video = QPushButton("Обработка видеофайла")
        btn_video.setStyleSheet("font-size: 18px; padding: 15px;")
        btn_video.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        
        btn_camera = QPushButton("Работа с камерой в реальном времени")
        btn_camera.setStyleSheet("font-size: 18px; padding: 15px;")
        btn_camera.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))
        
        layout.addWidget(title)
        layout.addWidget(btn_video)
        layout.addWidget(btn_camera)
        layout.addStretch()
        
        screen.setLayout(layout)
        self.stacked_widget.addWidget(screen)

    def init_video_screen(self):
        screen = QWidget()
        layout = QVBoxLayout()
        
        btn_back = QPushButton("<= Назад в меню")
        btn_back.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        
        self.lbl_input = QLabel("Видео не выбрано")
        btn_select_in = QPushButton("Выбрать входное видео")
        btn_select_in.clicked.connect(self.select_input_video)
        
        self.lbl_output = QLabel("Путь сохранения не выбран")
        btn_select_out = QPushButton("Выбрать путь для сохранения")
        btn_select_out.clicked.connect(self.select_output_video)
        
        self.btn_process = QPushButton("Начать обработку")
        self.btn_process.setStyleSheet("background-color: green; color: white; padding: 10px;")
        self.btn_process.clicked.connect(self.start_video_processing)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        
        layout.addWidget(btn_back)
        layout.addWidget(QLabel("<h2>Режим обработки видео</h2>"))
        layout.addWidget(btn_select_in)
        layout.addWidget(self.lbl_input)
        layout.addWidget(btn_select_out)
        layout.addWidget(self.lbl_output)
        layout.addWidget(self.btn_process)
        layout.addWidget(self.progress_bar)
        layout.addStretch()
        
        screen.setLayout(layout)
        self.stacked_widget.addWidget(screen)

    def init_camera_screen(self):
        screen = QWidget()
        layout = QVBoxLayout()
        
        btn_back = QPushButton("<= Назад в меню")
        btn_back.clicked.connect(self.stop_camera_and_back)
        
        self.combo_cameras = QComboBox()
        self.detect_cameras()
        
        self.btn_toggle_cam = QPushButton("Запустить камеру")
        self.btn_toggle_cam.clicked.connect(self.toggle_camera)
        
        self.lbl_video_feed = QLabel("Здесь будет видеопоток")
        self.lbl_video_feed.setAlignment(Qt.AlignCenter)
        self.lbl_video_feed.setStyleSheet("background-color: black; color: white;")
        self.lbl_video_feed.setMinimumSize(640, 480)
        
        layout.addWidget(btn_back)
        layout.addWidget(QLabel("<h2>Режим реального времени</h2>"))
        
        cam_layout = QHBoxLayout()
        cam_layout.addWidget(QLabel("Выберите камеру:"))
        cam_layout.addWidget(self.combo_cameras)
        cam_layout.addWidget(self.btn_toggle_cam)
        
        layout.addLayout(cam_layout)
        layout.addWidget(self.lbl_video_feed)
        
        screen.setLayout(layout)
        self.stacked_widget.addWidget(screen)

    def select_input_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите видео", "", "Video files (*.mp4 *.avi *.mov)")
        if file_path:
            self.lbl_input.setText(file_path)

    def select_output_video(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить видео как", "output.mp4", "Video files (*.mp4)")
        if file_path:
            self.lbl_output.setText(file_path)

    def start_video_processing(self):
        in_path = self.lbl_input.text()
        out_path = self.lbl_output.text()
        
        if not os.path.exists(in_path) or out_path == "Путь сохранения не выбран":
            QMessageBox.warning(self, "Предупреждение", "Выберите корректные пути файлов!")
            return
            
        self.btn_process.setEnabled(False)
        self.btn_process.setText("Обработка...")
        
        self.video_thread = VideoWorker(in_path, out_path, self.model, self.calib_data)
        self.video_thread.progress_changed.connect(self.progress_bar.setValue)
        self.video_thread.finished.connect(self.on_video_finished)
        self.video_thread.start()

    def on_video_finished(self, path):
        self.btn_process.setEnabled(True)
        self.btn_process.setText("Начать обработку")
        QMessageBox.information(self, "Готово", f"Видео успешно сохранено:\n{path}")

    def detect_cameras(self):
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                self.combo_cameras.addItem(f"Камера {i}", i)
                cap.release()

    def toggle_camera(self):
        if hasattr(self, 'cam_thread') and self.cam_thread.isRunning():
            self.cam_thread.stop()
            self.btn_toggle_cam.setText("Запустить камеру")
            self.lbl_video_feed.setText("Камера остановлена")
            self.lbl_video_feed.setStyleSheet("background-color: black; color: white;")
        else:
            cam_idx = self.combo_cameras.currentData()
            self.cam_thread = CameraWorker(cam_idx, self.model, self.calib_data)
            self.cam_thread.frame_ready.connect(self.update_video_feed)
            self.cam_thread.start()
            self.btn_toggle_cam.setText("Остановить камеру")

    def update_video_feed(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        self.lbl_video_feed.setPixmap(QPixmap.fromImage(p))

    def stop_camera_and_back(self):
        if hasattr(self, 'cam_thread') and self.cam_thread.isRunning():
            self.cam_thread.stop()
            self.btn_toggle_cam.setText("Запустить камеру")
        self.stacked_widget.setCurrentIndex(0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())