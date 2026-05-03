import sys
import json
import os
import numpy as np
import cv2
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QPushButton, QLineEdit, QVBoxLayout, QHBoxLayout,
    QGroupBox, QScrollArea, QMessageBox, QFileDialog,
    QGridLayout, QDialog, QComboBox
)
from PyQt5.QtCore import Qt, QPoint, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QPalette

DEFAULT_GRID_STEP     = 0.5
DEFAULT_MAX_DISTANCE  = 10.0
DEFAULT_CAMERA_HEIGHT = 0.5
DISPLAY_H             = 800

class CameraPreview(QDialog):
    frameCaptured = pyqtSignal(np.ndarray)

    def __init__(self, camera_index: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Предпросмотр камеры")
        self.setFixedSize(900, 700)

        self.current_frame = None

        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        layout = QVBoxLayout(self)

        self.lbl_feed = QLabel()
        self.lbl_feed.setAlignment(Qt.AlignCenter)
        self.lbl_feed.setStyleSheet("background: #111;")
        layout.addWidget(self.lbl_feed, stretch=1)

        btn_snap = QPushButton("📸  Сделать снимок")
        btn_snap.setFixedHeight(44)
        btn_snap.setStyleSheet(self._btn("#27ae60"))
        btn_snap.clicked.connect(self._capture)
        layout.addWidget(btn_snap)

        self.timer = QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(30)

    def _update(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        self.current_frame = frame.copy()  
        display = frame.copy()
        h, w = display.shape[:2]
        cx = w // 2
        cv2.line(display, (cx, 0), (cx, h), (0, 255, 0), 2)
        rgb  = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.lbl_feed.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.lbl_feed.width(), self.lbl_feed.height(),
                Qt.KeepAspectRatio
            )
        )

    def _capture(self):
        if self.current_frame is not None:
            self.frameCaptured.emit(self.current_frame.copy())
        self.close()

    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        super().closeEvent(event)

    @staticmethod
    def _btn(color):
        return (
            f"QPushButton {{ background-color: {color}; color: white;"
            f" border-radius: 5px; font-weight: bold; padding: 6px; }}"
        )

class ImageLabel(QLabel):
    pointSelected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.base_scale  = 1.0
        self.zoom_factor = 1.0
        self.points      = []
        self.pending_y   = None
        self._pixmap_src = None
        self.locked      = False

        self.dragging        = False
        self.last_mouse_pos  = QPoint()

        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)

    def set_source_pixmap(self, pixmap: QPixmap, base_scale: float):
        self._pixmap_src = pixmap
        self.base_scale  = base_scale
        self.zoom_factor = 1.0
        self._update_display_size()

    def _get_scale(self):
        return self.base_scale * self.zoom_factor

    def _update_display_size(self):
        if self._pixmap_src:
            s = self._get_scale()
            self.setFixedSize(
                int(self._pixmap_src.width()  * s),
                int(self._pixmap_src.height() * s)
            )
            self.update()

    def _scroll_area(self):
        p = self.parent()
        while p:
            if isinstance(p, QScrollArea):
                return p
            p = p.parent()
        return None

    def wheelEvent(self, event):
        if not self._pixmap_src:
            return
        cursor_pos = event.pos()
        old_zoom   = self.zoom_factor

        if event.angleDelta().y() > 0:
            self.zoom_factor *= 1.15
        else:
            self.zoom_factor /= 1.15

        self.zoom_factor = max(0.5, min(self.zoom_factor, 20.0))
        if old_zoom == self.zoom_factor:
            return

        ratio = self.zoom_factor / old_zoom
        self._update_display_size()

        sa = self._scroll_area()
        if sa:
            delta = cursor_pos * (ratio - 1)
            sa.horizontalScrollBar().setValue(
                int(sa.horizontalScrollBar().value() + delta.x()))
            sa.verticalScrollBar().setValue(
                int(sa.verticalScrollBar().value() + delta.y()))

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.dragging       = True
            self.last_mouse_pos = event.globalPos()
            self.setCursor(Qt.ClosedHandCursor)
            return

        if not self.locked and event.button() == Qt.LeftButton and self._pixmap_src:
            s  = self._get_scale()
            cx = self.width() // 2
            if abs(event.x() - cx) <= 40:
                self.pending_y = int(event.y() / s)
                self.update()
                self.pointSelected.emit(self.pending_y)

    def mouseMoveEvent(self, event):
        if self.dragging:
            curr  = event.globalPos()
            delta = curr - self.last_mouse_pos
            self.last_mouse_pos = curr
            sa = self._scroll_area()
            if sa:
                sa.horizontalScrollBar().setValue(
                    sa.horizontalScrollBar().value() - delta.x())
                sa.verticalScrollBar().setValue(
                    sa.verticalScrollBar().value() - delta.y())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.dragging = False
            self.setCursor(Qt.ArrowCursor if self.locked else Qt.CrossCursor)

    def paintEvent(self, event):
        if self._pixmap_src is None:
            return
        painter = QPainter(self)
        s = self._get_scale()
        painter.drawPixmap(0, 0, self.width(), self.height(), self._pixmap_src)

        if not self.locked:
            cx = self.width() // 2
            painter.setPen(QPen(QColor(0, 255, 0, 120), 2))
            painter.drawLine(cx, 0, cx, self.height())

            for dist, y_orig in self.points:
                y_disp = int(y_orig * s)
                painter.setPen(QPen(Qt.white, 1))
                painter.setBrush(QColor(255, 0, 0))
                painter.drawEllipse(QPoint(cx, y_disp), 6, 6)
                painter.setPen(QPen(QColor(255, 150, 150)))
                painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
                painter.drawText(cx + 15, y_disp + 5, f"{dist}м")

            if self.pending_y is not None:
                y_disp = int(self.pending_y * s)
                painter.setBrush(QColor(255, 165, 0))
                painter.drawEllipse(QPoint(cx, y_disp), 7, 7)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Калибровка камеры")
        self.image      = None
        self.pixmap_orig = None
        self.points     = []
        self.calibrated = False

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(15, 15, 15, 15)

        self.img_label = ImageLabel()
        self.img_label.pointSelected.connect(self.on_point_selected)

        self.container = QWidget()
        cont_layout    = QGridLayout(self.container)
        cont_layout.addWidget(self.img_label, 0, 0, Qt.AlignCenter)
        self.container.setStyleSheet("background-color: #1a1a1a;")

        scroll = QScrollArea()
        scroll.setWidget(self.container)
        scroll.setWidgetResizable(True)
        root.addWidget(scroll, stretch=4)

        ctrl = QVBoxLayout()
        root.addLayout(ctrl, stretch=1)

        btn_load = QPushButton("📁 Загрузить фото")
        btn_load.setFixedHeight(40)
        btn_load.clicked.connect(self.load_image_dialog)
        btn_load.setStyleSheet(self._btn_style("#444"))
        ctrl.addWidget(btn_load)

        grp_cam = QGroupBox("Камера")
        grp_cam.setStyleSheet(self._grp_style())
        cam_layout = QVBoxLayout(grp_cam)

        self.combo_cam = QComboBox()
        self._detect_cameras()
        cam_layout.addWidget(self.combo_cam)

        btn_cam = QPushButton("📷 Сделать снимок с камеры")
        btn_cam.setStyleSheet(self._btn_style("#555"))
        btn_cam.clicked.connect(self.open_camera_preview)
        cam_layout.addWidget(btn_cam)

        ctrl.addWidget(grp_cam)

        grp_params = QGroupBox("Параметры")
        grp_params.setStyleSheet(self._grp_style())
        params_layout = QGridLayout(grp_params)
        params_layout.addWidget(QLabel("Высота (м):"), 0, 0)
        self.edit_h = QLineEdit(str(DEFAULT_CAMERA_HEIGHT))
        params_layout.addWidget(self.edit_h, 0, 1)
        params_layout.addWidget(QLabel("Шаг (м):"), 1, 0)
        self.edit_step = QLineEdit(str(DEFAULT_GRID_STEP))
        params_layout.addWidget(self.edit_step, 1, 1)
        params_layout.addWidget(QLabel("Макс. (м):"), 2, 0)
        self.edit_max = QLineEdit(str(DEFAULT_MAX_DISTANCE))
        params_layout.addWidget(self.edit_max, 2, 1)
        ctrl.addWidget(grp_params)

        self.grp_add = QGroupBox("1. Точка")
        self.grp_add.setStyleSheet(self._grp_style())
        add_layout = QVBoxLayout(self.grp_add)
        self.lbl_sel = QLabel("Не выбрана")
        self.lbl_sel.setStyleSheet("color: #ffa500; font-weight: bold;")
        add_layout.addWidget(self.lbl_sel)
        row_d = QHBoxLayout()
        row_d.addWidget(QLabel("D (м):"))
        self.input_dist = QLineEdit()
        self.input_dist.returnPressed.connect(self.add_point)
        row_d.addWidget(self.input_dist)
        add_layout.addLayout(row_d)
        self.btn_add = QPushButton("Добавить")
        self.btn_add.setEnabled(False)
        self.btn_add.clicked.connect(self.add_point)
        self.btn_add.setStyleSheet(self._btn_style("#27ae60"))
        add_layout.addWidget(self.btn_add)
        ctrl.addWidget(self.grp_add)

        self.grp_pts = QGroupBox("2. Список")
        self.grp_pts.setStyleSheet(self._grp_style())
        pts_layout = QVBoxLayout(self.grp_pts)
        self.lbl_pts = QLabel("Пусто")
        self.lbl_pts.setWordWrap(True)
        pts_layout.addWidget(self.lbl_pts)
        self.btn_undo = QPushButton("Удалить последнюю")
        self.btn_undo.clicked.connect(self.undo_point)
        self.btn_undo.setStyleSheet(self._btn_style("#2980b9"))
        pts_layout.addWidget(self.btn_undo)
        ctrl.addWidget(self.grp_pts)

        self.btn_build = QPushButton("РАССЧИТАТЬ")
        self.btn_build.setEnabled(False)
        self.btn_build.setFixedHeight(50)
        self.btn_build.clicked.connect(self.handle_build_click)
        self.btn_build.setStyleSheet(self._btn_style("#8e44ad"))
        ctrl.addWidget(self.btn_build)

        # График
        self.plot_label = QLabel()
        self.plot_label.setAlignment(Qt.AlignCenter)
        ctrl.addWidget(self.plot_label)

        ctrl.addStretch()

    def _detect_cameras(self):
        self.combo_cam.clear()
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                self.combo_cam.addItem(f"Камера {i}", i)
                cap.release()
        if self.combo_cam.count() == 0:
            self.combo_cam.addItem("Камеры не найдены", -1)

    def open_camera_preview(self):
        idx = self.combo_cam.currentData()
        if idx == -1:
            QMessageBox.warning(self, "Предупреждение", "Камеры не найдены.")
            return
        preview = CameraPreview(idx, parent=self)
        preview.frameCaptured.connect(self._on_frame_captured)
        preview.exec_()

    def _on_frame_captured(self, frame: np.ndarray):
        self._set_image(frame)

    def load_image_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выбрать фото", "", "Images (*.jpg *.png *.jpeg)"
        )
        if path:
            img = cv2.imread(path)
            if img is not None:
                self._set_image(img)

    def _set_image(self, img: np.ndarray):
        self.image = img
        h, w = img.shape[:2]
        base_scale = DISPLAY_H / h if h > DISPLAY_H else 1.0

        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        self.pixmap_orig = QPixmap.fromImage(qimg)

        self.img_label.set_source_pixmap(self.pixmap_orig, base_scale)
        self.img_label.locked = False
        self.img_label.setCursor(Qt.CrossCursor)

        # Сброс состояния
        self.btn_build.setText("РАССЧИТАТЬ")
        self.calibrated = False
        self.points     = []
        self.img_label.points = self.points
        self.plot_label.clear()
        self.grp_add.setVisible(True)
        self.grp_pts.setVisible(True)
        self._refresh_ui()

    def on_point_selected(self, y_orig: int):
        self.lbl_sel.setText(f"Y: {y_orig}px")
        self.btn_add.setEnabled(True)
        self.input_dist.setFocus()

    def add_point(self):
        if self.img_label.pending_y is None:
            return
        try:
            d = float(self.input_dist.text().replace(',', '.'))
            if d <= 0:
                raise ValueError
        except ValueError:
            return

        self.points.append((d, self.img_label.pending_y))
        self.img_label.points    = self.points
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
        self.lbl_pts.setText("\n".join(lines) if lines else "Пусто")
        self.btn_undo.setEnabled(len(self.points) > 0)
        self.btn_build.setEnabled(len(self.points) >= 3)
        self.img_label.update()

    def handle_build_click(self):
        if not self.calibrated:
            self.build_calibration()
        else:
            self.calibrated = False
            self.img_label.locked = False
            self.img_label.setCursor(Qt.CrossCursor)
            self.img_label.set_source_pixmap(
                self.pixmap_orig, self.img_label.base_scale
            )
            self.btn_build.setText("РАССЧИТАТЬ")
            self.plot_label.clear()
            self.grp_add.setVisible(True)
            self.grp_pts.setVisible(True)

    def build_calibration(self):
        try:
            cam_h = float(self.edit_h.text().replace(',', '.'))
            step  = float(self.edit_step.text().replace(',', '.'))
            max_d = float(self.edit_max.text().replace(',', '.'))
        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Проверьте параметры")
            return

        dists_arr = np.array([p[0] for p in self.points])
        y_arr     = np.array([p[1] for p in self.points])
        h_img, w_img = self.image.shape[:2]

        try:
            popt, _ = curve_fit(
                lambda d, fh, yh: fh / d + yh,
                dists_arr, y_arr,
                p0=[500.0, h_img * 0.3]
            )
            fh, y_horizon = popt
        except Exception:
            QMessageBox.critical(self, "Ошибка", "Ошибка построения модели")
            return

        result   = self.image.copy()
        cx       = w_img // 2
        grid_data = {}
        distances = np.arange(step, max_d + step, step)

        for d in distances:
            y_c = int(fh / d + y_horizon)
            if 0 <= y_c < h_img:
                grid_data[str(round(d, 1))] = int(y_c)
                pts = []
                f_px = fh / cam_h
                for x_off in range(-w_img // 2, w_img // 2, 20):
                    angle = np.arctan(x_off / f_px)
                    y_adj = int(fh / (d * np.cos(angle)) + y_horizon)
                    if 0 <= y_adj < h_img:
                        pts.append([cx + x_off, y_adj])
                if len(pts) > 1:
                    cv2.polylines(
                        result,
                        [np.array(pts, np.int32)],
                        False, (0, 0, 255), 2
                    )
                    cv2.putText(
                        result, f"{d:.1f}m",
                        (20, y_c - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                    )

        cv2.imwrite("calibration_grid.jpg", result)

        data = {
            "fh":             fh,
            "y_horizon":      y_horizon,
            "camera_height_m": cam_h,
            "image_height":   h_img,
            "focal_length_px": fh / cam_h,
            "grid":           grid_data
        }
        with open("calibration.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # График
        plt.figure(figsize=(4, 3))
        d_plot = np.linspace(min(dists_arr) * 0.8, max_d, 100)
        plt.plot(d_plot, fh / d_plot + y_horizon, "b-", label="Модель")
        plt.scatter(dists_arr, y_arr, c="r", s=20)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("calibration_curve.png", dpi=80)
        plt.close()

        self.calibrated       = True
        self.img_label.locked = True
        self.img_label.setCursor(Qt.ArrowCursor)

        rgb   = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        q_res = QImage(rgb.data, w_img, h_img, rgb.strides[0], QImage.Format_RGB888)
        self.img_label.set_source_pixmap(
            QPixmap.fromImage(q_res), self.img_label.base_scale
        )

        self.plot_label.setPixmap(QPixmap("calibration_curve.png"))
        self.btn_build.setText("РАССЧИТАТЬ ПОВТОРНО")
        self.grp_add.setVisible(False)
        self.grp_pts.setVisible(False)

        QMessageBox.information(
            self, "Готово",
            f"Калибровка сохранена в calibration.json\n"
            f"fh = {fh:.1f}  |  y_horizon = {y_horizon:.1f}\n"
            f"focal = {fh / cam_h:.1f} px"
        )

    @staticmethod
    def _btn_style(color):
        return (
            f"QPushButton {{ background-color: {color}; color: white;"
            f" border-radius: 5px; font-weight: bold; border: none; padding: 5px; }}"
            f"QPushButton:hover {{ background-color: {color}cc; }}"
            f"QPushButton:disabled {{ background-color: #333; color: #666; }}"
        )

    @staticmethod
    def _grp_style():
        return (
            "QGroupBox { font-weight: bold; color: #bbb; border: 1px solid #444;"
            " margin-top: 10px; padding-top: 10px; border-radius: 5px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; }"
        )

if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps,    True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    pal = QPalette()
    pal.setColor(QPalette.Window,     QColor(45, 45, 45))
    pal.setColor(QPalette.WindowText, Qt.white)
    pal.setColor(QPalette.Base,       QColor(30, 30, 30))
    pal.setColor(QPalette.Text,       Qt.white)
    app.setPalette(pal)

    win = MainWindow()
    win.resize(1300, 900)
    win.show()
    sys.exit(app.exec_())