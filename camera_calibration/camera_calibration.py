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
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QPalette

# ─── Настройки ───────────────────────────────────────────
IMAGE_PATH    = "calibration.jpg"
OUTPUT_JSON   = "calibration.json"
GRID_STEP     = 0.5
MAX_DISTANCE  = 10.0
CAMERA_HEIGHT = 0.5
DISPLAY_H     = 800 
# ─────────────────────────────────────────────────────────

class ImageLabel(QLabel):
    """Виджет изображения с исправленным зумом и центрированием."""
    pointSelected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.base_scale = 1.0
        self.zoom_factor = 1.0
        self.points = []
        self.pending_y = None
        self._pixmap_src = None
        
        self.dragging = False
        self.last_mouse_pos = QPoint()
        
        # Важно: выравнивание самого контента внутри QLabel
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)

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

    def _get_scroll_area(self):
        p = self.parent()
        while p:
            if isinstance(p, QScrollArea):
                return p
            p = p.parent()
        return None

    def wheelEvent(self, event):
        if not self._pixmap_src:
            return

        old_zoom = self.zoom_factor
        old_scale = self._get_current_total_scale()
        cursor_pos = event.pos()
        
        # 1. Изменяем зум
        angle = event.angleDelta().y()
        if angle > 0:
            self.zoom_factor *= 1.15
        else:
            self.zoom_factor /= 1.15
        
        # Ограничения
        self.zoom_factor = max(0.5, min(self.zoom_factor, 15.0))
        
        # ИСПРАВЛЕНИЕ 1: Если масштаб не изменился (достигли лимита), ничего не делаем
        if old_zoom == self.zoom_factor:
            return

        new_scale = self._get_current_total_scale()
        self._update_display_size()

        # 2. Корректировка прокрутки
        scroll_area = self._get_scroll_area()
        if scroll_area:
            scale_ratio = new_scale / old_scale
            new_cursor_pos = cursor_pos * scale_ratio
            delta = new_cursor_pos - cursor_pos
            
            h_bar = scroll_area.horizontalScrollBar()
            v_bar = scroll_area.verticalScrollBar()
            
            h_bar.setValue(int(h_bar.value() + delta.x()))
            v_bar.setValue(int(v_bar.value() + delta.y()))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._pixmap_src:
            s = self._get_current_total_scale()
            # Учитываем, что при AlignCenter координаты могут иметь смещение, 
            # но так как мы используем setFixedSize равный размеру картинки,
            # event.x() всегда будет корректным относительно края виджета.
            curr_w = self.width()
            cx = curr_w // 2
            
            if abs(event.x() - cx) <= 40:
                self.pending_y = int(event.y() / s)
                self.update()
                self.pointSelected.emit(self.pending_y)
        
        elif event.button() == Qt.RightButton:
            self.dragging = True
            self.last_mouse_pos = event.globalPos()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self.dragging:
            current_pos = event.globalPos()
            delta = current_pos - self.last_mouse_pos
            self.last_mouse_pos = current_pos
            
            scroll_area = self._get_scroll_area()
            if scroll_area:
                scroll_area.horizontalScrollBar().setValue(
                    scroll_area.horizontalScrollBar().value() - delta.x()
                )
                scroll_area.verticalScrollBar().setValue(
                    scroll_area.verticalScrollBar().value() - delta.y()
                )
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.dragging = False
            self.setCursor(Qt.CrossCursor)

    def paintEvent(self, event):
        if self._pixmap_src is None:
            return
        
        painter = QPainter(self)
        s = self._get_current_total_scale()
        
        curr_w = int(self._pixmap_src.width() * s)
        curr_h = int(self._pixmap_src.height() * s)
        
        # Рисуем изображение
        painter.drawPixmap(0, 0, curr_w, curr_h, self._pixmap_src)

        cx = curr_w // 2
        # Линия
        painter.setPen(QPen(QColor(0, 255, 0, 120), 2))
        painter.drawLine(cx, 0, cx, curr_h)

        # Точки
        for dist, y_orig in self.points:
            y_disp = int(y_orig * s)
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.setBrush(QColor(255, 0, 0))
            painter.drawEllipse(QPoint(cx, y_disp), 6, 6)
            painter.setPen(QPen(QColor(255, 150, 150)))
            painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
            painter.drawText(cx + 15, y_disp + 5, f"{dist}м")

        if self.pending_y is not None:
            y_disp = int(self.pending_y * s)
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.setBrush(QColor(255, 165, 0))
            painter.drawEllipse(QPoint(cx, y_disp), 7, 7)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Инструмент калибровки камеры")
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

        self.img_label = ImageLabel()
        self.img_label.pointSelected.connect(self.on_point_selected)
        
        scroll = QScrollArea()
        scroll.setWidget(self.img_label)
        scroll.setWidgetResizable(False)
        
        # ИСПРАВЛЕНИЕ 2: Центрирование фото в области просмотра
        scroll.setAlignment(Qt.AlignCenter) 
        
        scroll.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")
        root.addWidget(scroll, stretch=4)

        ctrl = QVBoxLayout()
        ctrl.setSpacing(15)
        root.addLayout(ctrl, stretch=1)

        info = QLabel(
            "🖱 Колесо: Зум в точку\n"
            "🖱 Зажать ПКМ: Перемещение\n"
            "🖱 ЛКМ по линии: Выбор Y\n"
            "⌨ Введите дистанцию и Enter"
        )
        info.setStyleSheet("color: #aaa; font-size: 13px; line-height: 140%;")
        ctrl.addWidget(info)

        # Стили и группы (без изменений)
        grp_add = QGroupBox("1. Добавить точку")
        grp_add.setStyleSheet(self._grp_style())
        add_layout = QVBoxLayout(grp_add)
        self.lbl_selected = QLabel("Точка не выбрана")
        self.lbl_selected.setStyleSheet("color: #ffa500; font-weight: bold; font-size: 13px;")
        add_layout.addWidget(self.lbl_selected)
        row = QHBoxLayout(); row.addWidget(QLabel("Дист. (м):"))
        self.input_dist = QLineEdit()
        self.input_dist.setPlaceholderText("напр. 2.5")
        self.input_dist.setFixedHeight(35)
        self.input_dist.returnPressed.connect(self.add_point)
        row.addWidget(self.input_dist)
        add_layout.addLayout(row)
        self.btn_add = QPushButton("Добавить")
        self.btn_add.setEnabled(False); self.btn_add.setFixedHeight(40)
        self.btn_add.clicked.connect(self.add_point)
        self.btn_add.setStyleSheet(self._btn_style("#27ae60"))
        add_layout.addWidget(self.btn_add)
        ctrl.addWidget(grp_add)

        grp_pts = QGroupBox("2. Список точек")
        grp_pts.setStyleSheet(self._grp_style())
        pts_layout = QVBoxLayout(grp_pts)
        self.lbl_points = QLabel("Список пуст")
        self.lbl_points.setStyleSheet("color: #ddd; font-size: 12px;"); self.lbl_points.setWordWrap(True)
        pts_layout.addWidget(self.lbl_points)
        self.btn_undo = QPushButton("Отменить")
        self.btn_undo.setEnabled(False); self.btn_undo.clicked.connect(self.undo_point)
        self.btn_undo.setStyleSheet(self._btn_style("#2980b9"))
        pts_layout.addWidget(self.btn_undo)
        ctrl.addWidget(grp_pts)

        self.btn_build = QPushButton("РАССЧИТАТЬ")
        self.btn_build.setEnabled(False); self.btn_build.setFixedHeight(55)
        self.btn_build.clicked.connect(self.build_and_save)
        self.btn_build.setStyleSheet(self._btn_style("#8e44ad"))
        ctrl.addWidget(self.btn_build)
        ctrl.addStretch()

    def _load_image(self):
        self.image = cv2.imread(IMAGE_PATH)
        if self.image is None:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить {IMAGE_PATH}")
            return
        h, w = self.image.shape[:2]
        base_scale = DISPLAY_H / h if h > DISPLAY_H else 1.0
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        self.img_label.set_source_pixmap(QPixmap.fromImage(qimg), base_scale)

    def on_point_selected(self, y_orig: int):
        self.lbl_selected.setText(f"Выбран Y: {y_orig}px")
        self.btn_add.setEnabled(True)
        self.input_dist.setFocus()

    def add_point(self):
        if self.img_label.pending_y is None: return
        try:
            dist = float(self.input_dist.text().replace(',', '.'))
            if dist <= 0: raise ValueError
        except:
            QMessageBox.warning(self, "Ошибка", "Введите число > 0")
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
        lines = [f"• {d}м → {y}px" for d, y in sorted(self.points)]
        self.lbl_points.setText("\n".join(lines) if lines else "Список пуст")
        self.btn_undo.setEnabled(len(self.points) > 0)
        self.btn_build.setEnabled(len(self.points) >= 3)
        self.img_label.update()

    def _fit_model(self):
        distances = np.array([p[0] for p in self.points])
        y_pixels  = np.array([p[1] for p in self.points])
        h = self.image.shape[0]
        try:
            popt, _ = curve_fit(lambda d, fh, yh: fh / d + yh, distances, y_pixels, p0=[500.0, h * 0.3])
            return popt
        except:
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
                grid[str(round(d, 1))] = int(y_c)
                pts = []
                for x_off in range(-w // 2, w // 2, 15):
                    angle = np.arctan(x_off / (fh / CAMERA_HEIGHT))
                    y_adj = int(fh / (d / np.cos(angle)) + y_horizon)
                    if 0 <= y_adj < h: pts.append([cx + x_off, y_adj])
                if len(pts) > 1:
                    cv2.polylines(result, [np.array(pts, np.int32)], False, (0, 255, 255), 2)
                    cv2.putText(result, f"{d}m", (20, y_c - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.imwrite("calibration_grid.jpg", result)
        data = {"fh": fh, "y_horizon": y_horizon, "camera_height_m": CAMERA_HEIGHT, "grid": grid}
        with open(OUTPUT_JSON, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        QMessageBox.information(self, "Успех", "Калибровка завершена.")

    @staticmethod
    def _btn_style(color):
        return f"QPushButton {{ background-color: {color}; color: white; border-radius: 6px; font-weight: bold; border: none; }} QPushButton:hover {{ opacity: 0.8; }} QPushButton:disabled {{ background-color: #333; color: #666; }}"

    @staticmethod
    def _grp_style():
        return "QGroupBox { font-weight: bold; color: #bbb; border: 1px solid #444; margin-top: 12px; padding-top: 15px; border-radius: 5px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; }"


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(45, 45, 45))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(30, 30, 30))
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(60, 60, 60))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.Highlight, QColor(142, 68, 173))
    app.setPalette(palette)
    win = MainWindow()
    win.resize(1280, 850)
    win.show()
    sys.exit(app.exec_())