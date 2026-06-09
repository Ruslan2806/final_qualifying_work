import os
import sys
import json
import time
import subprocess
from pathlib import Path
import logging

os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False

import cv2
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QComboBox, QStackedWidget, 
    QMessageBox, QProgressBar, QDialog, QFormLayout, QDoubleSpinBox, 
    QCheckBox, QFrame, QSizePolicy, QLineEdit, QGroupBox, QScrollArea, 
    QGridLayout
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QUrl, QPoint, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor, QFont, QPainter, QPen
from PyQt5.QtMultimedia import QSoundEffect

os.environ['YOLO_VERBOSE'] = 'False'
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
logging.getLogger('ultralytics').setLevel(logging.WARNING)

BASE_PATH        = Path(__file__).resolve().parent
CALIBRATION_PATH = BASE_PATH / "camera_calibration" / "calibration.json"
MODEL_PATH       = BASE_PATH / "neural_networks" / "yolov12" / "yolov12.pt"
SETTINGS_PATH    = BASE_PATH / "settings.json"

CAR_ICON_PATH      = BASE_PATH / "materials" / "car.png"
DANGER_SOUND_PATH  = BASE_PATH / "materials" / "danger.wav"
WARNING_SOUND_PATH = BASE_PATH / "materials" / "warning.wav"
FONT_PATH          = BASE_PATH / "materials" / "font.ttf"
TELEMETRY_LOG_PATH = BASE_PATH / "telemetry_log.json"

CAR_ICON = None
if CAR_ICON_PATH.exists():
    CAR_ICON = cv2.imread(str(CAR_ICON_PATH), cv2.IMREAD_UNCHANGED)

DEFAULT_GRID_STEP     = 0.5
DEFAULT_MAX_DISTANCE  = 10.0
DEFAULT_CAMERA_HEIGHT = 0.5
DISPLAY_H             = 800

MAX_TRACK_AGE   = 90   
TIME_TO_LOG_CALMAN = 10.0