"""合成动作样本：把一张真人正面照加工成「做了动作」的序列。

为什么需要
----------
自动化测试需要一个**已知正确答案**的输入。真人做动作无法自动化，所以用图像处理模拟：

    · 张嘴 —— 在嘴内画一个暗色椭圆（真实口腔就是暗腔）
    · 抖动 —— 整帧小幅平移（模拟「有人举着照片手在抖」，这是**负样本**）

局限（必须记住）
----------------
这些都是**外观近似**，不是真实肌肉运动。它们能证明「判据与阈值量级合理、
照片类输入过不去」，不能代替真人在摄像头前的验收。
"""
import os

import cv2
import numpy as np

from backend.recognizer import recognizer

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHOTO = os.path.join(BASE, "data", "test", "real", "lena.jpg")


def load_frame(width=448):
    """读入测试照片并缩放到前端采集的宽度（448，q0.62 那一套参数）。"""
    raw = cv2.imdecode(np.fromfile(PHOTO, dtype=np.uint8), cv2.IMREAD_COLOR)
    if raw is None:
        raise SystemExit(f"读不到测试照片: {PHOTO}")
    h = int(raw.shape[0] * width / raw.shape[1])
    return cv2.resize(raw, (width, h), interpolation=cv2.INTER_CUBIC)


class Face:
    """一张可反复加工的正面脸。坐标换算与 `recognizer.align_face` 保持一致。"""

    def __init__(self, width=448):
        self.img = load_frame(width)
        faces = recognizer.detect(self.img)
        if not faces:
            raise SystemExit("测试照片里检测不到人脸")
        self.face = max(
            faces,
            key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
        )
        self.bbox = self.face["bbox"]
        self.pts, self.crop = recognizer.landmarks(self.img, self.bbox)
        x1, y1, x2, y2 = self.bbox
        self.cx, self.cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        # 与 recognizer.align_face 的缩放系数一致：192 / (脸长边 × 1.5)
        self.sc = 192 / (max(x2 - x1, y2 - y1) * 1.5)

    def to_img(self, p):
        """192 对齐空间坐标 → 原图坐标。"""
        return ((p[0] - 96) / self.sc + self.cx, (p[1] - 96) / self.sc + self.cy)

    def mouth_open(self, strength=1.0):
        """嘴内画暗色口腔 —— 模拟张嘴。strength 控制张开幅度（1.0 = 正常）。"""
        f = self.img.copy()
        inner = np.array([self.to_img(p) for p in self.pts[list(range(62, 72))]])
        ctr = inner.mean(0)
        axes = (
            max(3, int(np.ptp(inner[:, 0]) / 2 * 0.85 * strength)),
            max(3, int(np.ptp(inner[:, 1]) / 2 * 2.2 * strength)),
        )
        cv2.ellipse(f, (int(ctr[0]), int(ctr[1])), axes, 0, 0, 360, (28, 20, 18), -1)
        return f

    def shaken(self, n=22, amp=6):
        """整帧小幅平移 —— 模拟手持照片。负样本：动作判据必须不通过。"""
        out = []
        for i in range(n):
            dx = amp * np.sin(i * 0.9)
            dy = amp * np.cos(i * 1.3) * 0.6
            m = np.float32([[1, 0, dx], [0, 1, dy]])
            out.append(cv2.warpAffine(self.img, m, (self.img.shape[1], self.img.shape[0]),
                                      borderMode=cv2.BORDER_REPLICATE))
        return out


def sequence(face, kind=None, n=22, spans=None, start=5, length=8, **kw):
    """生成一段 n 帧序列。

    参数
    ----
    kind  : None（静止）| "mouth"（张嘴）| "mouth2"（张两次）
    spans : 自定义 (起点, 长度) 列表，覆盖 kind 的默认节奏
    start / length : 默认动作发生的位置与持续帧数（22 帧 × 100ms ≈ 2.2 秒）

    例：张两次 = spans=[(3, 4), (9, 5)]，即第 3~6 帧与第 9~13 帧各张一次。
    """
    if kind is None and spans is None:
        return [face.img] * n
    if spans is None:
        if kind == "mouth":
            spans = [(start, length)]
        elif kind == "mouth2":
            spans = [(3, 4), (9, 5)]
        else:
            raise ValueError(f"未知动作 {kind}")
    acted = face.mouth_open(**kw)
    frames = [face.img] * n
    for s, ln in spans:
        for i in range(max(0, s), min(n, s + ln)):
            frames[i] = acted
    return frames
