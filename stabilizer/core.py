"""手ブレ補正エンジン(GUI 非依存)

処理の流れ
  1. analyze()      : 動画を1回読み、フレーム間のカメラの動き (dx, dy, 回転, 拡大率) を推定
  2. Stabilization  : 動きを積算した「カメラの軌跡」を平滑化し、各フレームの補正行列を計算
  3. export()       : 元の解像度で補正をかけ、ffmpeg で書き出し(音声も保持)
"""
from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, List, Optional

import cv2
import numpy as np

ANALYSIS_WIDTH = 960      # 動き解析用の縮小幅(速度と精度のバランス)
PREVIEW_WIDTH = 960       # プレビュー用キャッシュの幅

ProgressCB = Optional[Callable[[int, int], None]]
CancelCB = Optional[Callable[[], bool]]


class Cancelled(Exception):
    pass


# --------------------------------------------------------------------------------------
# 解析
# --------------------------------------------------------------------------------------
@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int


@dataclass
class Analysis:
    info: VideoInfo
    motions: np.ndarray                 # (N, 4) フレーム i-1 → i の動き [dx, dy, 回転rad, log拡大率]
    previews: List[bytes] = field(default_factory=list)  # プレビュー用 JPEG
    preview_size: tuple = (0, 0)        # (w, h)

    @property
    def trajectory(self) -> np.ndarray:
        """カメラの軌跡(動きの積算)"""
        return np.cumsum(self.motions, axis=0)


def probe(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"動画を開けませんでした: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return VideoInfo(path, w, h, fps, n)


def _estimate_motion(prev_gray: np.ndarray, gray: np.ndarray) -> Optional[np.ndarray]:
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=500, qualityLevel=0.01,
                                  minDistance=15, blockSize=3)
    if pts is None or len(pts) < 10:
        return None
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None,
                                          winSize=(21, 21), maxLevel=3)
    if nxt is None:
        return None
    good = st.ravel() == 1
    if good.sum() < 10:
        return None
    m, _ = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC,
                                       ransacReprojThreshold=2.0, maxIters=2000,
                                       confidence=0.99)
    return m


def analyze(path: str, progress: ProgressCB = None, cancelled: CancelCB = None,
            keep_previews: bool = True) -> Analysis:
    info = probe(path)
    cap = cv2.VideoCapture(path)
    # 実際に読めたフレームの解像度を使う(回転メタデータ適用後の向き)
    ok, frame = cap.read()
    if not ok:
        raise IOError("フレームを読み込めませんでした(未対応のコーデックの可能性があります)")
    h, w = frame.shape[:2]
    info.width, info.height = w, h

    scale = min(1.0, ANALYSIS_WIDTH / w)
    aw, ah = int(round(w * scale)), int(round(h * scale))
    # 長尺はプレビューを小さくしてメモリを節約
    pw_target = PREVIEW_WIDTH if info.frame_count <= 15000 else 640
    pscale = min(1.0, pw_target / w)
    pw, ph = int(round(w * pscale)), int(round(h * pscale))
    center = np.array([aw / 2.0, ah / 2.0])

    motions = [np.zeros(4)]
    previews: List[bytes] = []
    prev_gray = None
    idx = 0
    total = max(info.frame_count, 1)
    while ok:
        small = cv2.resize(frame, (aw, ah), interpolation=cv2.INTER_AREA)
        if keep_previews:
            pv = small if (pw, ph) == (aw, ah) else cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
            previews.append(cv2.imencode(".jpg", pv, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            m = _estimate_motion(prev_gray, gray)
            if m is None:
                motions.append(np.zeros(4))
            else:
                a = m[:, :2]
                # 画面中心まわりの平行移動に変換: p' - c = A (p - c) + (A c + t - c)
                t = a @ center + m[:, 2] - center
                da = math.atan2(m[1, 0], m[0, 0])
                ds = math.log(max(1e-6, math.hypot(m[0, 0], m[1, 0])))
                motions.append(np.array([t[0] / scale, t[1] / scale, da, ds]))
        prev_gray = gray
        idx += 1
        if progress and idx % 5 == 0:
            progress(idx, max(total, idx))
        if cancelled and cancelled():
            cap.release()
            raise Cancelled()
        ok, frame = cap.read()
    cap.release()
    info.frame_count = idx
    if progress:
        progress(idx, idx)
    return Analysis(info, np.array(motions), previews, (pw, ph))


# --------------------------------------------------------------------------------------
# 補正計算
# --------------------------------------------------------------------------------------
BORDER_MODES = {
    "black": cv2.BORDER_CONSTANT,
    "reflect": cv2.BORDER_REFLECT_101,
    "replicate": cv2.BORDER_REPLICATE,
}


@dataclass
class Settings:
    mode: str = "smooth"           # "smooth"(なめらか) / "tripod"(三脚固定)
    smoothness_sec: float = 0.75   # 平滑化の強さ(ガウシアンのσ、秒)
    correct_rotation: bool = True
    correct_zoom: bool = False
    auto_crop: bool = True
    manual_zoom: float = 1.10      # auto_crop=False のときの拡大率
    max_auto_zoom: float = 1.6
    border: str = "black"


def gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or len(x) < 3:
        return x.copy()
    r = int(math.ceil(sigma * 3))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    n = len(x)
    out = np.empty_like(x, dtype=float)
    for c in range(x.shape[1]):
        col = x[:, c]
        # 端は「直線的に延長」して端点の引っ張られを防ぐ
        pad_l = 2 * col[0] - col[1:r + 1][::-1] if n > r else np.full(r, col[0])
        pad_r = 2 * col[-1] - col[-r - 1:-1][::-1] if n > r else np.full(r, col[-1])
        if len(pad_l) < r:
            pad_l = np.concatenate([np.full(r - len(pad_l), pad_l[0] if len(pad_l) else col[0]), pad_l])
        if len(pad_r) < r:
            pad_r = np.concatenate([pad_r, np.full(r - len(pad_r), pad_r[-1] if len(pad_r) else col[-1])])
        padded = np.concatenate([pad_l, col, pad_r])
        out[:, c] = np.convolve(padded, k, mode="valid")
    return out


class Stabilization:
    """解析結果 + 設定 → 各フレームの補正行列"""

    def __init__(self, analysis: Analysis, settings: Settings):
        self.analysis = analysis
        self.settings = settings
        self.recompute()

    def recompute(self):
        s = self.settings
        a = self.analysis
        traj = a.trajectory
        mask = np.array([1.0, 1.0, 1.0 if s.correct_rotation else 0.0, 1.0 if s.correct_zoom else 0.0])
        if s.mode == "tripod":
            smooth = np.tile(np.median(traj, axis=0), (len(traj), 1))
        else:
            smooth = gaussian_smooth(traj, s.smoothness_sec * a.info.fps)
        self.trajectory = traj
        self.smoothed = traj + (smooth - traj) * mask    # 補正後のカメラの軌跡
        self.corrections = self.smoothed - traj          # 各フレームに加える補正量
        self.required_zoom = self._required_zoom()
        if s.auto_crop:
            self.zoom = min(self.required_zoom, s.max_auto_zoom)
        else:
            self.zoom = s.manual_zoom

    # 揺れの指標: 軌跡の2階差分(フレームごとの加速度)のRMS [px]
    def shake_metrics(self):
        def rms(p):
            if len(p) < 3:
                return 0.0
            d2 = np.diff(p[:, :2], n=2, axis=0)
            return float(np.sqrt((d2 ** 2).sum(axis=1).mean()))
        before = rms(self.trajectory)
        after = rms(self.smoothed)
        reduction = 0.0 if before < 1e-9 else max(0.0, 1.0 - after / before)
        return before, after, reduction

    def _required_zoom(self) -> float:
        """どのフレームでも黒い縁が出ない最小の拡大率"""
        w, h = self.analysis.info.width, self.analysis.info.height
        d = self.corrections
        tx, ty, th, s = d[:, 0], d[:, 1], d[:, 2], np.exp(d[:, 3])
        cos, sin = np.cos(th), np.sin(th)
        corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])

        def covered(z):
            ok = np.ones(len(d), bool)
            for qx, qy in corners:
                ux, uy = qx / z - tx, qy / z - ty
                # v = R^T u / s
                vx = (cos * ux + sin * uy) / s
                vy = (-sin * ux + cos * uy) / s
                ok &= (np.abs(vx) <= w / 2 + 1e-6) & (np.abs(vy) <= h / 2 + 1e-6)
            return ok

        lo = np.ones(len(d))
        hi = np.full(len(d), 4.0)
        need = ~covered(lo)
        for _ in range(30):
            mid = (lo + hi) / 2
            c = covered(mid)
            hi = np.where(c, mid, hi)
            lo = np.where(c, lo, mid)
        z = np.where(need, hi, 1.0)
        return float(z.max()) * 1.002 if len(z) else 1.0

    def matrix(self, i: int, scale: float = 1.0) -> np.ndarray:
        """フレーム i の補正アフィン行列 (2x3)。scale はプレビュー縮小率"""
        w, h = self.analysis.info.width * scale, self.analysis.info.height * scale
        tx, ty, th, ls = self.corrections[i]
        z = self.zoom
        s = math.exp(ls)
        c, sn = math.cos(th), math.sin(th)
        a = z * s * np.array([[c, -sn], [sn, c]])
        center = np.array([w / 2, h / 2])
        t = center + z * np.array([tx, ty]) * scale - a @ center
        return np.hstack([a, t[:, None]])

    def apply(self, frame: np.ndarray, i: int, scale: float = 1.0) -> np.ndarray:
        h, w = frame.shape[:2]
        m = self.matrix(i, scale)
        border = BORDER_MODES.get(self.settings.border, cv2.BORDER_CONSTANT)
        return cv2.warpAffine(frame, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=border)

    def crop_polygon(self, i: int, scale: float = 1.0) -> np.ndarray:
        """補正後に残る範囲を元フレーム上の四角形として返す(表示用)"""
        w, h = self.analysis.info.width * scale, self.analysis.info.height * scale
        inv = cv2.invertAffineTransform(self.matrix(i, scale))
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
        return corners @ inv[:, :2].T + inv[:, 2]


# --------------------------------------------------------------------------------------
# 書き出し
# --------------------------------------------------------------------------------------
def find_ffmpeg() -> Optional[str]:
    import os
    # Homebrew などシステムの ffmpeg を優先、なければ imageio-ffmpeg 同梱版
    for p in (shutil.which("ffmpeg"), "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if p and os.path.exists(p):
            return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


CODECS = {
    # key: (表示名, 拡張子, ffmpeg 映像引数, 必要なエンコーダ)
    "h264": ("H.264 (MP4・高画質)", ".mp4",
             ["-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p"], "libx264"),
    "h264_vt": ("H.264 (MP4・Mac ハードウェア高速)", ".mp4",
                ["-c:v", "h264_videotoolbox", "-q:v", "70", "-pix_fmt", "yuv420p"], "h264_videotoolbox"),
    "hevc": ("H.265 / HEVC (MP4・小さいファイル)", ".mp4",
             ["-c:v", "libx265", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-tag:v", "hvc1"],
             "libx265"),
    "hevc_vt": ("H.265 / HEVC (MP4・Mac ハードウェア高速)", ".mp4",
                ["-c:v", "hevc_videotoolbox", "-q:v", "65", "-tag:v", "hvc1", "-pix_fmt", "yuv420p"],
                "hevc_videotoolbox"),
    "prores": ("ProRes 422 HQ (MOV・編集用)", ".mov",
               ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"], "prores_ks"),
}


def available_codecs(ffmpeg: Optional[str] = None) -> List[str]:
    ffmpeg = ffmpeg or find_ffmpeg()
    if not ffmpeg:
        return []
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True,
                             text=True, timeout=20).stdout
    except Exception:
        return []
    return [k for k, v in CODECS.items() if f" {v[3]} " in out]


def export(stab: Stabilization, out_path: str, codec: str = "h264",
           start: int = 0, end: Optional[int] = None,
           progress: ProgressCB = None, cancelled: CancelCB = None) -> None:
    info = stab.analysis.info
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg が見つかりません。`pip install imageio-ffmpeg` または `brew install ffmpeg` を実行してください。")
    end = info.frame_count if end is None else min(end, info.frame_count)
    if end <= start:
        raise ValueError("書き出し範囲が空です")
    w, h = info.width, info.height
    # 偶数サイズに(4:2:0 の要件)
    ow, oh = w - (w % 2), h - (h % 2)
    fps = Fraction(info.fps).limit_denominator(1001)
    t0 = start / info.fps
    dur = (end - start) / info.fps
    ext = out_path.lower().rsplit(".", 1)[-1]
    audio = ["-c:a", "pcm_s16le"] if ext == "mov" else ["-c:a", "aac", "-b:a", "320k"]
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}", "-r", f"{fps.numerator}/{fps.denominator}",
           "-i", "-",
           "-ss", f"{t0:.6f}", "-t", f"{dur:.6f}", "-i", info.path,
           "-map", "0:v:0", "-map", "1:a:0?",
           *CODECS[codec][2], *audio,
           "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
           "-movflags", "+faststart", "-shortest", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    cap = cv2.VideoCapture(info.path)
    try:
        for _ in range(start):          # 正確なフレーム位置にするため先頭から読み飛ばす
            if not cap.grab():
                break
            if cancelled and cancelled():
                raise Cancelled()
        for i in range(start, end):
            ok, frame = cap.read()
            if not ok:
                break
            out = stab.apply(frame, i)
            if (ow, oh) != (w, h):
                out = out[:oh, :ow]
            try:
                proc.stdin.write(np.ascontiguousarray(out).tobytes())
            except BrokenPipeError:
                break
            if progress and (i - start) % 3 == 0:
                progress(i - start + 1, end - start)
            if cancelled and cancelled():
                raise Cancelled()
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", "replace")
        if proc.wait() != 0:
            raise RuntimeError("書き出しに失敗しました:\n" + err[-2000:])
        if progress:
            progress(end - start, end - start)
    except BaseException:
        proc.kill()
        proc.wait()
        import os
        try:
            os.remove(out_path)
        except OSError:
            pass
        raise
    finally:
        cap.release()
