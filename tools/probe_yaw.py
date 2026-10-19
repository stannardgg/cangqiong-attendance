"""标定「转头」动作的可用指标（先量，再定阈值）。

为什么需要这个脚本
------------------
原计划用「眼睛开合比」判动作，实验证明关键点回归器对局部形变不敏感（见 probe_landmark.py）。
转头是**整体姿态变化**，情况不一样：关键点应当跟随大范围旋转。但「跟随程度」必须实测，
不能凭直觉定阈值 —— 定高了真人转不过，定低了照片用平面旋转就能蒙混。

方法
----
对同一张正面照做**绕 Y 轴的透视变换**（K·R·K⁻¹，等价于把脸绕竖轴转过去），
这比「横向缩放」更接近真实的偏转效果（近侧变大、远侧压缩）。
每个角度重新检测 + 取关键点，计算 4 个候选指标，看哪个随角度单调、区分度最大。

候选指标（都在 192 对齐空间、且已用双眼连线把旋转转正）
    1. asym       双眼中点相对左右轮廓边缘的不对称度
    2. eye_asym   两只眼宽度的不对称度
    3. mouth_shift 嘴中心相对双眼中点的水平偏移（按脸宽归一）
    4. spare      脸部两侧「多余宽度」，同上但用嘴角而不是眼
"""
import os
import sys

import cv2
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from backend.recognizer import recognizer  # noqa: E402

PHOTO = os.path.join(BASE, "data", "test", "real", "lena.jpg")
EYE_A = list(range(33, 43))
EYE_B = list(range(87, 97))
MOUTH_IN = list(range(62, 72))
CONTOUR = list(range(0, 33))


def yaw_warp(img: np.ndarray, deg: float, center=None) -> np.ndarray:
    """绕 Y 轴旋转 deg 度的透视变换（把平面脸近似成立体头在转）。

    center 必须传**人脸中心**：绕画面中心转会把人脸整体平移出画面，
    大角度下就「检测不到人脸」了，那测到的是模拟误差，不是判据的极限。
    """
    h, w = img.shape[:2]
    f = 1.2 * w
    cx, cy = (w / 2.0, h / 2.0) if center is None else (float(center[0]), float(center[1]))
    k = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], np.float64)
    t = np.deg2rad(deg)
    r = np.array([[np.cos(t), 0, np.sin(t)],
                  [0, 1, 0],
                  [-np.sin(t), 0, np.cos(t)]], np.float64)
    # 先把人脸中心挪到画面中心，转完再挪回原位
    pre = np.array([[1, 0, w / 2.0 - cx], [0, 1, h / 2.0 - cy], [0, 0, 1]], np.float64)
    post = np.array([[1, 0, cx - w / 2.0], [0, 1, cy - h / 2.0], [0, 0, 1]], np.float64)
    h_m = post @ (k @ r @ np.linalg.inv(k)) @ pre
    return cv2.warpPerspective(img, h_m, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def orient(pts: np.ndarray) -> np.ndarray:
    """用双眼连线把点集旋转到「两眼水平」，消掉平面内旋转的干扰。"""
    la = pts[EYE_A].mean(0)
    lb = pts[EYE_B].mean(0)
    ang = np.arctan2(lb[1] - la[1], lb[0] - la[0])
    c, s = np.cos(-ang), np.sin(-ang)
    r = np.array([[c, -s], [s, c]], np.float64)
    return (pts - (la + lb) / 2.0) @ r.T


def metrics(pts: np.ndarray) -> dict:
    p = orient(pts)
    ca = p[EYE_A]
    cb = p[EYE_B]
    ex = (ca[:, 0].mean() + cb[:, 0].mean()) / 2.0
    c = p[CONTOUR]
    xmin, xmax = float(c[:, 0].min()), float(c[:, 0].max())
    width = max(xmax - xmin, 1e-3)
    left, right = ex - xmin, xmax - ex
    wa = float(np.ptp(ca[:, 0]))
    wb = float(np.ptp(cb[:, 0]))
    mi = p[MOUTH_IN]
    mouth_x = float(mi[:, 0].mean())
    return {
        "asym": (left - right) / max(left + right, 1e-3),
        "eye_asym": (wa - wb) / max(wa + wb, 1e-3),
        "mouth_shift": (mouth_x - ex) / width,
    }


def main():
    raw = cv2.imdecode(np.fromfile(PHOTO, dtype=np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(raw, (448, int(raw.shape[0] * 448 / raw.shape[1])), interpolation=cv2.INTER_CUBIC)

    print(f"样本 {img.shape[1]}x{img.shape[0]}（与前端采集同宽）\n")
    print(f"{'角度':>6} | {'asym':>8} | {'eye_asym':>9} | {'mouth_shift':>12}")
    print("-" * 46)
    rows = []
    for deg in (-40, -30, -20, -10, 0, 10, 20, 30, 40):
        w = yaw_warp(img, deg)
        faces = recognizer.detect(w)
        if not faces:
            print(f"{deg:>6} | 检测不到人脸")
            continue
        face = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
        pts, _ = recognizer.landmarks(w, face["bbox"])
        m = metrics(pts)
        rows.append((deg, m))
        print(f"{deg:>6}° | {m['asym']:+.4f} | {m['eye_asym']:+.4f} | {m['mouth_shift']:+.4f}")

    if len(rows) >= 3:
        print("\n区分度（同一指标在 0° 与各角度之间的最大绝对差）：")
        base = dict(rows[[r[0] for r in rows].index(0)][1]) if any(r[0] == 0 for r in rows) else None
        if base:
            for k in ("asym", "eye_asym", "mouth_shift"):
                span = max(abs(m[k] - base[k]) for _, m in rows)
                print(f"  {k:<12} 最大偏离 {span:.4f}")
        print("\n趋势单调性（|值| 是否随角度增大）：")
        for k in ("asym", "eye_asym", "mouth_shift"):
            vals = [abs(m[k]) for deg, m in rows if abs(deg) >= 10]
            mono = all(b >= a - 0.01 for a, b in zip(vals, vals[1:]))
            print(f"  {k:<12} {[round(v, 3) for v in vals]}  单调={mono}")


if __name__ == "__main__":
    main()
