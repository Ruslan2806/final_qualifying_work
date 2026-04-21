import sys
import json
import numpy as np
import cv2
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QPushButton, QLineEdit, QVBoxLayout, QHBoxLayout,
    QGroupBox, QScrollArea, QMessageBox
)
from PyQt5.QtCore import Qt, QPoint, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont

# ─── Settings ────────────────────────────────────────────
IMAGE_PATH    = "calibration.jpg"
OUTPUT_JSON   = "calibration.json"
GRID_STEP     = 0.5
MAX_DISTANCE  = 10.0
CAMERA_HEIGHT = 0.5
DISPLAY_H     = 800  # Базовая высота для первого отображения
# ─────────────────────────────────────────────────────────

class ImageLabel(QLabel):
    """Кастомный QLabel с поддержкой зума и сигналом клика."""
    pointSelected = pyqtSignal(int)  # Сигнал передает Y координату в пикселях оригинала

    def __init__(self, parent=None):
        super().__init__(parent)
        self.base_scale = 1.0   # Масштаб, чтобы вписать фото в экран
        self.zoom_factor = 1.0  # Множитель зума от пользователя
        self.points = []        # [(dist_m, y_orig), ...]
        self.pending_y = None   # Y в координатах оригинального фото
        self._pixmap_src = None # Оригинальный QPixmap (полный размер)
        
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setCursor(Qt.CrossCursor)

    def set_source_pixmap(self, pixmap: QPixmap, base_scale: float):
        self._pixmap_src = pixmap
        self.base_scale = base_scale
        self.zoom_factor = 1.0
        self._update_display_size()

    def _get_current_total_scale(self):
        return self.base_scale * self.zoom_factor

    def _update_display_size(self):
        if self._pixmap_src:
            s = self._get_current_total_scale()
            new_w = int(self._pixmap_src.width() * s)
            new_h = int(self._pixmap_src.height() * s)
            self.setFixedSize(new_w, new_h)
            self.update()

    def wheelEvent(self, event):
        """Зум колесиком мыши."""
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_factor *= 1.1
        else:
            self.zoom_factor /= 1.1
        
        # Ограничения зума
        self.zoom_factor = max(0.5, min(self.zoom_factor, 10.0))
        self._update_display_size()

    def paintEvent(self, event):
        if self._pixmap_src is None:
            return
        
        painter = QPainter(self)
        s = self._get_current_total_scale()
        
        # Рисуем масштабированное изображение
        curr_w = int(self._pixmap_src.width() * s)
        curr_h = int(self._pixmap_src.height() * s)
        painter.drawPixmap(0, 0, curr_w, curr_h, self._pixmap_src)

        cx = curr_w // 2

        # Центральная линия
        painter.setPen(QPen(QColor(0, 255, 0, 150), 2))
        painter.drawLine(cx, 0, cx, curr_h)

        # Отрисовка подтвержденных точек
        for dist, y_orig in self.points:
            y_disp = int(y_orig * s)
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.setBrush(QColor(255, 0, 0))
            painter.drawEllipse(QPoint(cx, y_disp), 6, 6)
            painter.setPen(QPen(QColor(255, 100, 100)))
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            painter.drawText(cx + 15, y_disp + 5, f"{dist}m")

        # Оранжевая точка (выбранная, но не подтвержденная)
        if self.pending_y is not None:
            y_disp = int(self.pending_y * s)
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.setBrush(QColor(255, 165, 0))
            painter.drawEllipse(QPoint(cx, y_disp), 6, 6)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or self._pixmap_src is None:
            return
        
        s = self._get_current_total_scale()
        curr_w = int(self._pixmap_src.width() * s)
        cx = curr_w // 2
        
        # Проверка: клик вблизи центральной линии (с учетом масштаба)
        if abs(event.x() - cx) <= 30:
            self.pending_y = int(event.y() / s)
            self.update()
            self.pointSelected.emit(self.pending_y)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Calibration Tool")
        self.image = None
        self.points = []

        self._build_ui()
        self._load_image()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(20)
        root.setContentsMargins(15, 15, 15, 15)

        # ── Левая часть: Изображение ──────────────────────
        self.img_label = ImageLabel()
        # Соединяем сигнал со слотом
        self.img_label.pointSelected.connect(self.on_point_selected)
        
        scroll = QScrollArea()
        scroll.setWidget(self.img_label)
        scroll.setWidgetResizable(False) # Важно для работы зума
        scroll.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")
        root.addWidget(scroll, stretch=4)

        # ── Правая часть: Управление ──────────────────────
        ctrl = QVBoxLayout()
        ctrl.setSpacing(15)
        root.addLayout(ctrl, stretch=1)

        info = QLabel(
            "🖯 Mouse Wheel: Zoom\n"
            "🖱 Click green line to select Y\n"
            "⌨ Enter distance & press Add"
        )
        info.setStyleSheet("color: #888; font-size: 13px; line-height: 150%;")
        ctrl.addWidget(info)

        # Группа добавления
        grp_add = QGroupBox("1. Add Point")
        grp_add.setStyleSheet(self._grp_style())
        add_layout = QVBoxLayout(grp_add)

        self.lbl_selected = QLabel("No point selected")
        self.lbl_selected.setStyleSheet("color: #ffa500; font-weight: bold;")
        add_layout.addWidget(self.lbl_selected)

        row = QHBoxLayout()
        row.addWidget(QLabel("Dist (m):"))
        self.input_dist = QLineEdit()
        self.input_dist.setPlaceholderText("e.g. 2.0")
        self.input_dist.setFixedHeight(35)
        self.input_dist.returnPressed.connect(self.add_point)
        row.addWidget(self.input_dist)
        add_layout.addLayout(row)

        self.btn_add = QPushButton("Add Point")
        self.btn_add.setEnabled(False)
        self.btn_add.setFixedHeight(40)
        self.btn_add.clicked.connect(self.add_point)
        self.btn_add.setStyleSheet(self._btn_style("#27ae60"))
        add_layout.addWidget(self.btn_add)
        ctrl.addWidget(grp_add)

        # Группа списка точек
        grp_pts = QGroupBox("2. Points List")
        grp_pts.setStyleSheet(self._grp_style())
        pts_layout = QVBoxLayout(grp_pts)
        self.lbl_points = QLabel("Empty")
        self.lbl_points.setWordWrap(True)
        pts_layout.addWidget(self.lbl_points)
        
        self.btn_undo = QPushButton("Undo Last")
        self.btn_undo.setEnabled(False)
        self.btn_undo.clicked.connect(self.undo_point)
        self.btn_undo.setStyleSheet(self._btn_style("#2980b9"))
        pts_layout.addWidget(self.btn_undo)
        ctrl.addWidget(grp_pts)

        # Группа финализации
        self.btn_build = QPushButton("BUILD CALIBRATION")
        self.btn_build.setEnabled(False)
        self.btn_build.setFixedHeight(50)
        self.btn_build.clicked.connect(self.build_and_save)
        self.btn_build.setStyleSheet(self._btn_style("#8e44ad"))
        ctrl.addWidget(self.btn_build)
        
        ctrl.addStretch()

    def _load_image(self):
        self.image = cv2.imread(IMAGE_PATH)
        if self.image is None:
            QMessageBox.critical(self, "Error", f"Could not load {IMAGE_PATH}")
            return

        h, w = self.image.shape[:2]
        base_scale = DISPLAY_H / h if h > DISPLAY_H else 1.0
        
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        self.img_label.set_source_pixmap(pixmap, base_scale)

    # ── Слот для сигнала из ImageLabel ──────────────────
    def on_point_selected(self, y_orig: int):
        self.lbl_selected.setText(f"Selected Y: {y_orig}px")
        self.btn_add.setEnabled(True)
        self.input_dist.setFocus()

    def add_point(self):
        if self.img_label.pending_y is None: return
        try:
            dist = float(self.input_dist.text().replace(',', '.'))
            if dist <= 0: raise ValueError
        except:
            QMessageBox.warning(self, "Error", "Invalid distance")
            return

        self.points.append((dist, self.img_label.pending_y))
        self.img_label.points = self.points
        self.img_label.pending_y = None
        self.input_dist.clear()
        self.btn_add.setEnabled(False)
        self._refresh_ui()

    def undo_point(self):
        if self.points:
            self.points.pop()
            self._refresh_ui()

    def _refresh_ui(self):
        lines = [f"• {d}m → {y}px" for d, y in self.points]
        self.lbl_points.setText("\n".join(lines) if lines else "Empty")
        self.btn_undo.setEnabled(len(self.points) > 0)
        self.btn_build.setEnabled(len(self.points) >= 3)
        self.img_label.update()

    def _fit_model(self):
        distances = np.array([p[0] for p in self.points])
        y_pixels  = np.array([p[1] for p in self.points])
        h = self.image.shape[0]
        try:
            # Модель: y = (f*h)/d + y_horizon
            popt, _ = curve_fit(
                lambda d, fh, yh: fh / d + yh,
                distances, y_pixels,
                p0=[500.0, h * 0.3]
            )
            return popt # fh, y_horizon
        except Exception as e:
            QMessageBox.critical(self, "Math Error", f"Could not fit model: {e}")
            return None, None

    def build_and_save(self):
        fh, y_horizon = self._fit_model()
        if fh is None: return

        h, w = self.image.shape[:2]
        cx = w // 2
        dists = np.arange(GRID_STEP, MAX_DISTANCE + GRID_STEP, GRID_STEP)
        grid = {}
        result = self.image.copy()

        for d in dists:
            y_c = int(fh / d + y_horizon)
            if 0 <= y_c < h:
                grid[round(d, 1)] = y_c
                pts = []
                # Рисуем дугу (учитываем, что расстояние до боковых точек больше)
                for x_off in range(-w // 2, w // 2, 10):
                    f_px = fh / CAMERA_HEIGHT
                    angle = np.arctan(x_off / f_px)
                    y_adj = int(fh / (d / np.cos(angle)) + y_horizon)
                    if 0 <= y_adj < h:
                        pts.append([cx + x_off, y_adj])
                
                if len(pts) > 1:
                    cv2.polylines(result, [np.array(pts, np.int32)], False, (0, 255, 255), 2)
                    cv2.putText(result, f"{d}m", (20, y_c), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # Сохранение и показ
        cv2.imwrite("calibration_grid.jpg", result)
        
        # JSON
        data = {
            "fh": fh, "y_horizon": y_horizon, "camera_height": CAMERA_HEIGHT,
            "focal_length_px": fh / CAMERA_HEIGHT, "grid": grid
        }
        with open(OUTPUT_JSON, "w") as f:
            json.dump(data, f, indent=2)

        QMessageBox.information(self, "Success", f"Calibration saved to {OUTPUT_JSON}\nGrid image saved.")

    @staticmethod
    def _btn_style(color):
        return f"""
            QPushButton {{
                background-color: {color}; color: white; border-radius: 4px; 
                font-weight: bold; font-size: 14px;
            }}
            QPushButton:hover {{ background-color: white; color: {color}; border: 2px solid {color}; }}
            QPushButton:disabled {{ background-color: #444; color: #777; }}
        """

    @staticmethod
    def _grp_style():
        return "QGroupBox { font-weight: bold; color: #bbb; border: 1px solid #444; margin-top: 10px; padding: 10px; }"


if __name__ == "__main__":
    # Поддержка High DPI
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Темная тема
    from PyQt5.QtGui import QPalette
    p = QPalette()
    p.setColor(QPalette.Window, QColor(40, 40, 40))
    p.setColor(QPalette.WindowText, Qt.white)
    p.setColor(QPalette.Base, QColor(25, 25, 25))
    p.setColor(QPalette.Text, Qt.white)
    app.setPalette(p)

    win = MainWindow()
    win.resize(1200, 900)
    win.show()
    sys.exit(app.exec_())