"""手ブレ補正アプリ GUI (PySide6)"""
from __future__ import annotations

import os
import sys
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QColor, QImage, QKeySequence, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QProgressDialog,
                               QPushButton, QSizePolicy, QSlider, QSplitter, QToolBar, QVBoxLayout, QWidget)

from . import core

VIDEO_FILTER = "動画 (*.mp4 *.mov *.m4v *.mts *.m2ts *.avi *.mkv *.MP4 *.MOV *.MTS);;すべてのファイル (*)"


# --------------------------------------------------------------------------------------
# バックグラウンド処理
# --------------------------------------------------------------------------------------
class Worker(QObject):
    progress = Signal(int, int)
    done = Signal(object)
    failed = Signal(str)
    cancelled_sig = Signal()

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            res = self.fn(*self.args, progress=self.progress.emit, cancelled=lambda: self._cancel, **self.kwargs)
            self.done.emit(res)
        except core.Cancelled:
            self.cancelled_sig.emit()
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


def run_in_thread(parent, worker: Worker) -> QThread:
    th = QThread(parent)
    worker.moveToThread(th)
    th.started.connect(worker.run)
    for s in (worker.done, worker.failed, worker.cancelled_sig):
        s.connect(th.quit)
    th.finished.connect(th.deleteLater)
    th.start()
    return th


# --------------------------------------------------------------------------------------
# 映像表示
# --------------------------------------------------------------------------------------
class VideoView(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(480, 270)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#111; color:#888; font-size:16px;")
        self.setText("動画ファイルをここにドラッグ&ドロップ\nまたは「開く」から選択してください")
        self._img: Optional[QImage] = None

    def set_frame(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self._img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self._refresh()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._refresh()

    def _refresh(self):
        if self._img is not None:
            self.setPixmap(QPixmap.fromImage(self._img).scaled(self.size(), Qt.KeepAspectRatio,
                                                               Qt.SmoothTransformation))


class TrajectoryGraph(QWidget):
    """カメラの軌跡(補正前=灰色/補正後=色付き)を表示。クリックでシーク"""
    seek = Signal(int)
    LANES = [("横 (X)", 0, "#4fc3f7", 1.0), ("縦 (Y)", 1, "#81c784", 1.0), ("回転 (°)", 2, "#ffb74d", 57.2958)]

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(170)
        self.stab: Optional[core.Stabilization] = None
        self.pos = 0
        self.in_out = (0, 0)

    def set_data(self, stab, pos, in_out):
        self.stab, self.pos, self.in_out = stab, pos, in_out
        self.update()

    def _x(self, i, n):
        return 60 + (self.width() - 70) * i / max(1, n - 1)

    def mousePressEvent(self, e):
        self._seek(e)

    def mouseMoveEvent(self, e):
        self._seek(e)

    def _seek(self, e):
        if not self.stab:
            return
        n = len(self.stab.trajectory)
        i = int(round((e.position().x() - 60) / max(1, self.width() - 70) * (n - 1)))
        self.seek.emit(max(0, min(n - 1, i)))

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#1b1b1b"))
        if not self.stab:
            p.setPen(QColor("#666"))
            p.drawText(self.rect(), Qt.AlignCenter, "解析後にカメラの揺れのグラフが表示されます")
            return
        raw, sm = self.stab.trajectory, self.stab.smoothed
        n = len(raw)
        lane_h = (self.height() - 8) / len(self.LANES)
        # In/Out 範囲
        x0, x1 = self._x(self.in_out[0], n), self._x(self.in_out[1] - 1, n)
        p.fillRect(QRectF(x0, 0, max(1, x1 - x0), self.height()), QColor(255, 255, 255, 18))
        step = max(1, n // max(1, self.width()))
        idx = np.arange(0, n, step)
        for li, (name, ch, color, mul) in enumerate(self.LANES):
            top = 4 + li * lane_h
            r, s = raw[idx, ch] * mul, sm[idx, ch] * mul
            lo, hi = min(r.min(), s.min()), max(r.max(), s.max())
            if hi - lo < 1e-6:
                lo, hi = lo - 1, hi + 1
            pad = (hi - lo) * 0.1

            def y(v):
                return top + lane_h - 4 - (v - lo + pad) / (hi - lo + 2 * pad) * (lane_h - 8)

            p.setPen(QPen(QColor("#333"), 1))
            p.drawLine(QPointF(60, top + lane_h), QPointF(self.width(), top + lane_h))
            p.setPen(QColor("#aaa"))
            p.drawText(QRectF(2, top, 56, lane_h), Qt.AlignVCenter | Qt.AlignLeft, name)
            for series, pen in ((r, QPen(QColor("#777"), 1)), (s, QPen(QColor(color), 2))):
                path = QPainterPath()
                for k, v in enumerate(series):
                    pt = QPointF(self._x(idx[k], n), y(v))
                    path.moveTo(pt) if k == 0 else path.lineTo(pt)
                p.setPen(pen)
                p.drawPath(path)
        p.setPen(QPen(QColor("#ff5252"), 1.5))
        xp = self._x(self.pos, n)
        p.drawLine(QPointF(xp, 0), QPointF(xp, self.height()))
        p.setPen(QColor("#999"))
        p.drawText(QRectF(60, 0, self.width() - 66, 16), Qt.AlignRight, "灰: 補正前の揺れ  /  色: 補正後")


# --------------------------------------------------------------------------------------
# メインウィンドウ
# --------------------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("手ブレ補正 - Stabilizer")
        self.resize(1400, 900)
        self.setAcceptDrops(True)
        self.analysis: Optional[core.Analysis] = None
        self.stab: Optional[core.Stabilization] = None
        self.settings = core.Settings()
        self.pos = 0
        self.in_pt, self.out_pt = 0, 0
        self._thread = None
        self._worker = None
        self._dlg = None
        self._frame_cache = (-1, None)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        self._build_ui()
        self._set_enabled(False)

    # ---------------- UI ----------------
    def _build_ui(self):
        tb = QToolBar()
        tb.setMovable(False)
        self.addToolBar(tb)
        act_open = QAction("📂 開く", self, shortcut=QKeySequence.Open, triggered=self.open_dialog)
        self.act_export = QAction("💾 書き出し", self, shortcut=QKeySequence("Ctrl+E"), triggered=self.export_dialog)
        tb.addAction(act_open)
        tb.addAction(self.act_export)
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color:#888; padding-left:12px;")
        tb.addWidget(self.info_label)

        # 左: 映像 + グラフ + 再生操作
        self.view = VideoView()
        self.graph = TrajectoryGraph()
        self.graph.seek.connect(self.seek)

        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedWidth(44)
        self.btn_play.clicked.connect(self.toggle_play)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.seek)
        self.time_label = QLabel("00:00.00 / 00:00.00")
        self.time_label.setMinimumWidth(170)
        btn_in = QPushButton("開始点 [I]")
        btn_in.clicked.connect(self.set_in)
        btn_out = QPushButton("終了点 [O]")
        btn_out.clicked.connect(self.set_out)
        btn_clear = QPushButton("範囲解除")
        btn_clear.clicked.connect(self.clear_in_out)
        transport = QHBoxLayout()
        for w in (self.btn_play, self.slider, self.time_label, btn_in, btn_out, btn_clear):
            transport.addWidget(w)
        self._transport_widgets = [self.btn_play, self.slider, btn_in, btn_out, btn_clear]

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.view)
        splitter.addWidget(self.graph)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        left = QVBoxLayout()
        left.addWidget(splitter, 1)
        left.addLayout(transport)

        # 右: 設定パネル
        panel = QVBoxLayout()

        g_view = QGroupBox("表示")
        f = QFormLayout(g_view)
        self.cb_display = QComboBox()
        self.cb_display.addItems(["並べて比較(左: 元 / 右: 補正後)", "補正後のみ", "元動画のみ"])
        self.cb_display.currentIndexChanged.connect(self.refresh_frame)
        self.chk_cropframe = QCheckBox("元動画に切り出し枠を表示")
        self.chk_cropframe.setChecked(True)
        self.chk_cropframe.toggled.connect(self.refresh_frame)
        f.addRow(self.cb_display)
        f.addRow(self.chk_cropframe)
        panel.addWidget(g_view)

        g_st = QGroupBox("手ブレ補正")
        f = QFormLayout(g_st)
        self.cb_mode = QComboBox()
        self.cb_mode.addItems(["なめらか(手持ち・歩き撮り・パン)", "三脚固定(カメラを動かさない場面)"])
        self.cb_mode.currentIndexChanged.connect(self._settings_changed)
        f.addRow("モード", self.cb_mode)
        self.sl_strength = QSlider(Qt.Horizontal)
        self.sl_strength.setRange(1, 60)          # ×0.05 秒
        self.sl_strength.setValue(int(self.settings.smoothness_sec / 0.05))
        self.sl_strength.valueChanged.connect(self._settings_changed)
        self.lb_strength = QLabel()
        row = QHBoxLayout()
        row.addWidget(self.sl_strength)
        row.addWidget(self.lb_strength)
        f.addRow("補正の強さ", row)
        self.chk_rot = QCheckBox("回転の揺れも補正")
        self.chk_rot.setChecked(self.settings.correct_rotation)
        self.chk_rot.toggled.connect(self._settings_changed)
        self.chk_zoom = QCheckBox("前後の揺れ(拡大縮小)も補正")
        self.chk_zoom.setChecked(self.settings.correct_zoom)
        self.chk_zoom.toggled.connect(self._settings_changed)
        f.addRow(self.chk_rot)
        f.addRow(self.chk_zoom)
        panel.addWidget(g_st)

        g_crop = QGroupBox("切り出し(画面の縁の処理)")
        f = QFormLayout(g_crop)
        self.chk_autocrop = QCheckBox("自動で拡大して黒い縁を隠す")
        self.chk_autocrop.setChecked(True)
        self.chk_autocrop.toggled.connect(self._settings_changed)
        f.addRow(self.chk_autocrop)
        self.sp_maxzoom = QDoubleSpinBox()
        self.sp_maxzoom.setRange(100, 300)
        self.sp_maxzoom.setSuffix(" %")
        self.sp_maxzoom.setValue(self.settings.max_auto_zoom * 100)
        self.sp_maxzoom.valueChanged.connect(self._settings_changed)
        f.addRow("自動拡大の上限", self.sp_maxzoom)
        self.sp_zoom = QDoubleSpinBox()
        self.sp_zoom.setRange(100, 300)
        self.sp_zoom.setSingleStep(1)
        self.sp_zoom.setSuffix(" %")
        self.sp_zoom.setValue(self.settings.manual_zoom * 100)
        self.sp_zoom.valueChanged.connect(self._settings_changed)
        f.addRow("手動拡大率", self.sp_zoom)
        self.cb_border = QComboBox()
        self.cb_border.addItems(["黒", "鏡像で埋める", "端を引き伸ばす"])
        self.cb_border.currentIndexChanged.connect(self._settings_changed)
        f.addRow("縁の塗りつぶし", self.cb_border)
        panel.addWidget(g_crop)

        g_stat = QGroupBox("補正状況")
        v = QVBoxLayout(g_stat)
        self.stat_label = QLabel("—")
        self.stat_label.setWordWrap(True)
        self.stat_label.setTextFormat(Qt.RichText)
        v.addWidget(self.stat_label)
        panel.addWidget(g_stat)

        self.btn_export = QPushButton("💾 書き出し…")
        self.btn_export.setMinimumHeight(40)
        self.btn_export.clicked.connect(self.export_dialog)
        panel.addWidget(self.btn_export)
        panel.addStretch(1)
        help_lb = QLabel("ショートカット: スペース=再生/停止, ←/→=1フレーム移動, "
                         "Shift+←/→=1秒移動, I/O=開始点/終了点, ⌘O=開く, ⌘E=書き出し")
        help_lb.setWordWrap(True)
        help_lb.setStyleSheet("color:#888; font-size:11px;")
        panel.addWidget(help_lb)

        right = QWidget()
        right.setLayout(panel)
        right.setFixedWidth(340)
        self._panel = right

        central = QWidget()
        h = QHBoxLayout(central)
        lw = QWidget()
        lw.setLayout(left)
        h.addWidget(lw, 1)
        h.addWidget(right)
        self.setCentralWidget(central)
        self._update_strength_label()

    def _set_enabled(self, on: bool):
        self._panel.setEnabled(on)
        self.act_export.setEnabled(on)
        for w in self._transport_widgets:
            w.setEnabled(on)

    # ---------------- ファイル ----------------
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            self.load(urls[0].toLocalFile())

    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "動画を開く", os.path.expanduser("~/Movies"), VIDEO_FILTER)
        if path:
            self.load(path)

    def load(self, path: str):
        if self._thread is not None:
            return
        self.pause()
        try:
            info = core.probe(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "エラー", str(e))
            return
        dlg = QProgressDialog("手ブレを解析しています…\n(動画の長さによって数十秒〜数分かかります)", "キャンセル",
                              0, max(1, info.frame_count), self)
        dlg.setWindowTitle("解析中")
        self._start_task(Worker(core.analyze, path), dlg, self._on_analyzed, "解析エラー")

    # ---------------- バックグラウンド処理の管理(UI 操作は必ずメインスレッドで) ----------------
    def _start_task(self, worker: Worker, dlg: QProgressDialog, on_done, err_title: str):
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.canceled.connect(worker.cancel)
        self._dlg, self._on_done, self._err_title = dlg, on_done, err_title
        worker.progress.connect(self._task_progress)
        worker.done.connect(self._task_done)
        worker.failed.connect(self._task_failed)
        worker.cancelled_sig.connect(self._task_cancelled)
        self._worker = worker
        self._thread = run_in_thread(self, worker)

    def _task_end(self):
        self._thread = None
        self._worker = None
        if self._dlg:
            self._dlg.close()
            self._dlg.deleteLater()
            self._dlg = None

    @Slot(int, int)
    def _task_progress(self, i, n):
        if self._dlg:
            self._dlg.setMaximum(max(1, n))
            self._dlg.setValue(min(i, n))

    @Slot(object)
    def _task_done(self, res):
        self._task_end()
        self._on_done(res)

    @Slot(str)
    def _task_failed(self, msg):
        self._task_end()
        QMessageBox.critical(self, self._err_title, msg)

    @Slot()
    def _task_cancelled(self):
        self._task_end()

    def _on_analyzed(self, analysis: core.Analysis):
        self.analysis = analysis
        self.stab = core.Stabilization(analysis, self.settings)
        n = analysis.info.frame_count
        self.in_pt, self.out_pt = 0, n
        self.slider.blockSignals(True)
        self.slider.setRange(0, n - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        i = analysis.info
        self.info_label.setText(f"{os.path.basename(i.path)}   {i.width}×{i.height}  {i.fps:.3f}fps  {n}フレーム")
        self.setWindowTitle(f"手ブレ補正 - {os.path.basename(i.path)}")
        self._set_enabled(True)
        self._frame_cache = (-1, None)
        self.pos = 0
        self._settings_changed()

    # ---------------- 設定 ----------------
    def _update_strength_label(self):
        self.lb_strength.setText(f"{self.sl_strength.value() * 0.05:.2f} 秒")

    def _settings_changed(self, *_):
        s = self.settings
        s.mode = "tripod" if self.cb_mode.currentIndex() == 1 else "smooth"
        s.smoothness_sec = self.sl_strength.value() * 0.05
        s.correct_rotation = self.chk_rot.isChecked()
        s.correct_zoom = self.chk_zoom.isChecked()
        s.auto_crop = self.chk_autocrop.isChecked()
        s.max_auto_zoom = self.sp_maxzoom.value() / 100
        s.manual_zoom = self.sp_zoom.value() / 100
        s.border = ["black", "reflect", "replicate"][self.cb_border.currentIndex()]
        self.sl_strength.setEnabled(s.mode == "smooth")
        self.sp_zoom.setEnabled(not s.auto_crop)
        self.sp_maxzoom.setEnabled(s.auto_crop)
        self._update_strength_label()
        if not self.stab:
            return
        self.stab.recompute()
        self._update_stats()
        self.refresh_frame()

    def _update_stats(self):
        st = self.stab
        before, after, red = st.shake_metrics()
        need = st.required_zoom
        warn = ""
        if st.zoom + 1e-3 < need:
            warn = (f"<br><span style='color:#ff8a65'>⚠ 縁を完全に隠すには {need * 100:.0f}% の拡大が必要です。"
                    "一部のフレームで縁が見えます。補正を弱めるか拡大率を上げてください。</span>")
        elif need > 1.3:
            warn = "<br><span style='color:#ffd54f'>拡大率が大きめです(画質が低下します)。補正を弱めると改善します。</span>"
        self.stat_label.setText(
            f"揺れの量: <b>{before:.2f}px</b> → <b>{after:.2f}px</b><br>"
            f"揺れの低減率: <b style='font-size:16px;color:#69f0ae'>{red * 100:.1f}%</b><br>"
            f"拡大率(切り出し): <b>{st.zoom * 100:.1f}%</b>(必要: {need * 100:.1f}%){warn}")

    # ---------------- 再生 ----------------
    def _preview_frame(self, i: int) -> np.ndarray:
        if self._frame_cache[0] != i:
            buf = np.frombuffer(self.analysis.previews[i], np.uint8)
            self._frame_cache = (i, cv2.imdecode(buf, cv2.IMREAD_COLOR))
        return self._frame_cache[1]

    def refresh_frame(self, *_):
        if not self.stab:
            return
        i = self.pos
        frame = self._preview_frame(i)
        scale = frame.shape[1] / self.analysis.info.width
        mode = self.cb_display.currentIndex()
        if mode == 2:
            out = frame.copy()
        else:
            out = self.stab.apply(frame, i, scale)
        if mode in (0, 2) and self.chk_cropframe.isChecked():
            orig = frame.copy() if mode == 0 else out
            poly = self.stab.crop_polygon(i, scale).astype(np.int32)
            th = max(2, frame.shape[1] // 400)
            cv2.polylines(orig, [poly], True, (80, 220, 255), th, cv2.LINE_AA)
        else:
            orig = frame
        if mode == 0:
            gap = np.full((frame.shape[0], 8, 3), 17, np.uint8)
            out = np.hstack([orig, gap, out])
        elif mode == 2:
            out = orig
        self.view.set_frame(out)
        self.graph.set_data(self.stab, i, (self.in_pt, self.out_pt))
        fps = self.analysis.info.fps
        self.time_label.setText(f"{self._fmt(i / fps)} / {self._fmt(self.analysis.info.frame_count / fps)}")

    @staticmethod
    def _fmt(t):
        return f"{int(t // 60):02d}:{t % 60:05.2f}"

    def seek(self, i: int):
        if not self.analysis:
            return
        i = max(0, min(self.analysis.info.frame_count - 1, int(i)))
        self.pos = i
        if self.slider.value() != i:
            self.slider.blockSignals(True)
            self.slider.setValue(i)
            self.slider.blockSignals(False)
        self.refresh_frame()

    def toggle_play(self):
        if self.timer.isActive():
            self.pause()
        elif self.analysis:
            if self.pos >= self.out_pt - 1:
                self.seek(self.in_pt)
            self.timer.start(int(1000 / self.analysis.info.fps))
            self.btn_play.setText("❚❚")

    def pause(self):
        self.timer.stop()
        self.btn_play.setText("▶")

    def _tick(self):
        nxt = self.pos + 1
        if nxt >= self.out_pt:
            nxt = self.in_pt     # 範囲内をループ再生
        self.seek(nxt)

    def set_in(self):
        if self.analysis:
            self.in_pt = min(self.pos, self.out_pt - 1)
            self.refresh_frame()

    def set_out(self):
        if self.analysis:
            self.out_pt = max(self.pos + 1, self.in_pt + 1)
            self.refresh_frame()

    def clear_in_out(self):
        if self.analysis:
            self.in_pt, self.out_pt = 0, self.analysis.info.frame_count
            self.refresh_frame()

    def keyPressEvent(self, e):
        if not self.analysis:
            return super().keyPressEvent(e)
        k = e.key()
        step = int(round(self.analysis.info.fps)) if e.modifiers() & Qt.ShiftModifier else 1
        if k == Qt.Key_Space:
            self.toggle_play()
        elif k == Qt.Key_Left:
            self.pause()
            self.seek(self.pos - step)
        elif k == Qt.Key_Right:
            self.pause()
            self.seek(self.pos + step)
        elif k == Qt.Key_I:
            self.set_in()
        elif k == Qt.Key_O:
            self.set_out()
        else:
            super().keyPressEvent(e)

    # ---------------- 書き出し ----------------
    def export_dialog(self):
        if not self.stab or self._thread is not None:
            return
        self.pause()
        codecs = core.available_codecs()
        if not codecs:
            QMessageBox.critical(self, "エラー", "ffmpeg が見つかりません。\nターミナルで `pip install imageio-ffmpeg` "
                                 "または `brew install ffmpeg` を実行してください。")
            return
        from PySide6.QtWidgets import QInputDialog
        names = [core.CODECS[c][0] for c in codecs]
        name, ok = QInputDialog.getItem(self, "書き出し形式", "形式を選んでください:", names, 0, False)
        if not ok:
            return
        codec = codecs[names.index(name)]
        ext = core.CODECS[codec][1]
        src = self.analysis.info.path
        default = os.path.splitext(src)[0] + "_stabilized" + ext
        path, _ = QFileDialog.getSaveFileName(self, "書き出し先", default, f"動画 (*{ext})")
        if not path:
            return
        if not path.lower().endswith(ext):
            path += ext
        if os.path.abspath(path) == os.path.abspath(src):
            QMessageBox.warning(self, "エラー", "元の動画と同じファイルには書き出せません。")
            return
        dlg = QProgressDialog("書き出し中…", "キャンセル", 0, self.out_pt - self.in_pt, self)
        dlg.setWindowTitle("書き出し")
        self._export_path = path
        self._start_task(Worker(core.export, self.stab, path, codec, self.in_pt, self.out_pt), dlg,
                         self._on_exported, "書き出しエラー")

    def _on_exported(self, _):
        path = self._export_path
        box = QMessageBox(self)
        box.setWindowTitle("完了")
        box.setText(f"書き出しが完了しました。\n{path}")
        reveal = box.addButton("Finder で表示", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Ok)
        box.exec()
        if box.clickedButton() is reveal and sys.platform == "darwin":
            import subprocess
            subprocess.run(["open", "-R", path])

def main(argv=None):
    app = QApplication(sys.argv if argv is None else argv)
    app.setApplicationName("Stabilizer")
    app.setStyle("Fusion")
    from PySide6.QtGui import QPalette
    pal = QPalette()
    for role, c in [(QPalette.Window, "#262626"), (QPalette.WindowText, "#e0e0e0"), (QPalette.Base, "#1e1e1e"),
                    (QPalette.AlternateBase, "#2a2a2a"), (QPalette.Text, "#e0e0e0"), (QPalette.Button, "#333"),
                    (QPalette.ButtonText, "#e0e0e0"), (QPalette.Highlight, "#2979ff"),
                    (QPalette.HighlightedText, "#fff"), (QPalette.ToolTipBase, "#333"), (QPalette.ToolTipText, "#eee")]:
        pal.setColor(role, QColor(c))
    app.setPalette(pal)
    win = MainWindow()
    win.show()
    args = app.arguments()[1:]
    if args and os.path.exists(args[0]):
        QTimer.singleShot(100, lambda: win.load(args[0]))
    return app.exec()
