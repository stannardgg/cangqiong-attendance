"""临时探测脚本：2d106det 关键点可视化 + 各阶段推理耗时。

用途（一次性诊断，不属于运行时）：
  1. 确认 106 点的索引分区（眼部 / 嘴部 / 轮廓），避免凭记忆猜索引。
  2. 测量 CPU 上 detect / landmark / liveness / embedding 的单次耗时，
     用来决定「一次签到采集多少帧」的预算。
"""
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from backend.recognizer import recognizer  # noqa: E402

LMD = os.path.join(BASE, "models", "2d106det.onnx")


def landmarks(sess, img, bbox, size=192):
    """按 InsightFace 官方 Landmark 类的对齐方式：以人脸框中心 + 1.5 倍尺度裁剪到 192。"""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    scale = size / (max(w, h) * 1.5)
    m = cv2.getRotationMatrix2D((cx, cy), 0, scale)
    m[0, 2] += size / 2 - cx
    m[1, 2] += size / 2 - cy
    crop = cv2.warpAffine(img, m, (size, size), borderValue=0.0)
    blob = crop.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
    pred = sess.run(None, {sess.get_inputs()[0].name: blob})[0].reshape(-1, 2)
    pred = (pred + 1.0) * (size // 2)
    return pred, crop


def main():
    img_path = os.path.join(BASE, "data", "test", "real", "lena.jpg")
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    print("image", img.shape)

    t = time.perf_counter()
    faces = recognizer.detect(img)
    t_det = (time.perf_counter() - t) * 1000
    face = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
    print(f"detect x{len(faces)}: {t_det:.1f} ms  bbox={[round(v,1) for v in face['bbox']]}")

    sess = ort.InferenceSession(LMD, providers=["CPUExecutionProvider"])

    # 跑 5 次取均值
    times = []
    for _ in range(5):
        t = time.perf_counter()
        pts, crop = landmarks(sess, img, face["bbox"])
        times.append((time.perf_counter() - t) * 1000)
    print(f"landmark: {sum(times)/len(times):.1f} ms  pts={pts.shape}")

    t = time.perf_counter()
    recognizer.liveness_probs(img, face["bbox"])
    print(f"liveness: {(time.perf_counter()-t)*1000:.1f} ms")

    t = time.perf_counter()
    recognizer.get_embedding(img, face["kps"])
    print(f"embedding: {(time.perf_counter()-t)*1000:.1f} ms")

    # 可视化：放大 3 倍画编号
    canvas = cv2.resize(crop, (192 * 3, 192 * 3))
    for i, (x, y) in enumerate(pts):
        px, py = int(x * 3), int(y * 3)
        cv2.circle(canvas, (px, py), 2, (0, 255, 0), -1)
        cv2.putText(canvas, str(i), (px + 2, py - 2), cv2.FONT_HERSHEY_PLAIN,
                    0.55, (0, 0, 255), 1)
    # 中文路径：cv2.imwrite 会静默失败，必须用 imencode + 二进制写
    out = os.path.join(BASE, "output", "lmd_probe.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ok, buf = cv2.imencode(".png", canvas)
    with open(out, "wb") as f:
        f.write(buf.tobytes())
    print("saved", out, ok)

    # 打印各点的 (x, y) 便于定位分区
    for i in range(0, 106, 1):
        print(i, round(float(pts[i][0]), 1), round(float(pts[i][1]), 1))


if __name__ == "__main__":
    main()
