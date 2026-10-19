"""苍穹签到系统 — 人脸识别引擎（InsightFace + onnxruntime）。

模型（已下载到 models/）：
  - det_10g.onnx   ：SCRFD 人脸检测，输出人脸框 + 5 个关键点。
  - w600k_r50.onnx ：ArcFace 识别，输入 112x112 对齐人脸，输出 512 维特征。

实现参考 InsightFace 官方 SCRFD / ArcFace 推理流程，仅依赖
onnxruntime + numpy + opencv（headless），CPU 即可运行。
"""
import os

import cv2
import numpy as np
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

DET_MODEL = os.path.join(MODELS_DIR, "det_10g.onnx")
REC_MODEL = os.path.join(MODELS_DIR, "w600k_r50.onnx")
LIVENESS_MODEL = os.path.join(MODELS_DIR, "minifasnet_v2.onnx")  # 可选，静默活体检测
# 106 点关键点，用于「动作挑战」（眨眼 / 张嘴）。单帧活体无法区分照片与真人，
# 动作校验是唯一能拦住静态照片与预录视频的手段，见 backend/liveness.py。
LANDMARK_MODEL = os.path.join(MODELS_DIR, "2d106det.onnx")
LANDMARK_SIZE = 192        # 2d106det 固定输入 192x192
LANDMARK_MARGIN = 1.5      # 官方对齐尺度：人脸框最长边放大 1.5 倍后铺满 192

LIVENESS_SCALE = 2.7       # 活体模型人脸裁剪放大倍数（模型训练时的固定参数）
# MiniFASNetV2 输出 3 类 softmax，顺序固定为 [打印攻击, 真人, 翻拍攻击]，
# 「真人」固定在索引 1。取错索引（如 0）会把所有真人都判成假脸。
LIVENESS_LIVE_INDEX = 1
LIVENESS_THRESHOLD = 0.5   # 真人概率阈值（softmax 的 live 类概率）
# 输入标定系数（重要）：本仓库使用的 ONNX 导出存在约 255 倍的标定偏移——
# 若按官方 ToTensor 喂 [0,1]，网络会落入「全连接偏置主导」的饱和区，输出与输入几乎无关
# （实测真人帧与随机噪声得到相同结果）。实测喂原始像素值 [0,255] 时行为恢复正常：
# 真人 0.9999 / 打印攻击 0.0007 / 屏幕翻拍 0.0077，分离度约 1000 倍。
LIVENESS_INPUT_SCALE = 255.0

DET_SIZE = 640          # 检测输入尺寸
DET_THRESH = 0.5        # 检测置信度阈值
NMS_THRESH = 0.4        # 非极大值抑制 IoU 阈值
FEAT_STRIDES = [8, 16, 32]
NUM_ANCHORS = 2

# ArcFace 112x112 的 5 点标准模板（左眼、右眼、鼻尖、左嘴角、右嘴角）
ARC_SRC = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """两个特征向量的余弦相似度（-1 ~ 1，越大越像）。"""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.size == 0 or b.size == 0:
        return -1.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """SCRFD 框解码：anchor 中心 + 偏移 → [x1,y1,x2,y2]。"""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """SCRFD 关键点解码 → [N, 5, 2]。"""
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1).reshape((-1, 5, 2))


def _nms(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> list:
    """标准 NMS，返回保留的索引列表。"""
    if len(boxes) == 0:
        return []
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep


class Recognizer:
    """人脸检测 + 特征提取 + 比对。懒加载模型，首次调用时初始化。"""

    def __init__(self) -> None:
        self._det: ort.InferenceSession | None = None
        self._rec: ort.InferenceSession | None = None
        self._live: ort.InferenceSession | None = None
        self._lmd: ort.InferenceSession | None = None
        self._ready = False

    def is_ready(self) -> bool:
        return self._ready and self._det is not None and self._rec is not None

    def has_liveness(self) -> bool:
        """活体检测模型是否可用（可选增强）。"""
        return self._live is not None

    def has_landmark(self) -> bool:
        """关键点模型是否可用（动作挑战依赖它）。"""
        return self._lmd is not None

    def load(self) -> None:
        """加载检测与识别模型。模型文件缺失时抛出明确错误。"""
        if self.is_ready():
            return
        for path in (DET_MODEL, REC_MODEL):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"缺少模型文件 {os.path.basename(path)}，请将其放入 {MODELS_DIR} 目录。"
                )
        self._det = ort.InferenceSession(DET_MODEL, providers=["CPUExecutionProvider"])
        self._rec = ort.InferenceSession(REC_MODEL, providers=["CPUExecutionProvider"])
        # 活体模型为可选增强，缺失时不阻塞
        if os.path.exists(LIVENESS_MODEL):
            self._live = ort.InferenceSession(LIVENESS_MODEL, providers=["CPUExecutionProvider"])
        if os.path.exists(LANDMARK_MODEL):
            self._lmd = ort.InferenceSession(LANDMARK_MODEL, providers=["CPUExecutionProvider"])
        self._ready = True

    def detect(self, img_bgr: np.ndarray) -> list[dict]:
        """检测图像中所有人脸，返回 [{bbox（人脸在图像中的位置，例如左上角和右下角坐标）, kps, score}]，坐标为原图像素。"""
        self.load()
        h, w = img_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            img_bgr, 1.0 / 128.0, (DET_SIZE, DET_SIZE), (127.5, 127.5, 127.5), swapRB=True
        )
        input_name = self._det.get_inputs()[0].name
        net_outs = self._det.run(None, {input_name: blob})

        fmc = 3
        scores_list, bboxes_list, kpss_list = [], [], []
        ih, iw = blob.shape[2], blob.shape[3]
        for idx, stride in enumerate(FEAT_STRIDES):
            scores = net_outs[idx].ravel()
            bbox_preds = net_outs[idx + fmc] * stride
            kps_preds = net_outs[idx + fmc * 2] * stride
            fh, fw = ih // stride, iw // stride
            centers = np.stack(np.mgrid[:fh, :fw][::-1], axis=-1).astype(np.float32)
            centers = (centers * stride).reshape((-1, 2))
            if NUM_ANCHORS > 1:
                centers = np.stack([centers] * NUM_ANCHORS, axis=1).reshape((-1, 2))
            bboxes = _distance2bbox(centers, bbox_preds)
            kpss = _distance2kps(centers, kps_preds)
            pos = np.where(scores >= DET_THRESH)[0]
            scores_list.append(scores[pos])
            bboxes_list.append(bboxes[pos])
            kpss_list.append(kpss[pos])

        if not any(s.size for s in scores_list):
            return []
        scores = np.concatenate(scores_list)
        bboxes = np.concatenate(bboxes_list)
        kpss = np.concatenate(kpss_list)
        keep = _nms(bboxes, scores, NMS_THRESH)
        bboxes, scores, kpss = bboxes[keep], scores[keep], kpss[keep]

        # 缩放回原图坐标
        sx, sy = w / DET_SIZE, h / DET_SIZE
        bboxes = bboxes.copy()
        bboxes[:, [0, 2]] *= sx
        bboxes[:, [1, 3]] *= sy
        kpss = kpss.copy()
        kpss[:, :, 0] *= sx
        kpss[:, :, 1] *= sy
        bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, w)
        bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, h)

        return [
            {"bbox": bboxes[i].tolist(), "kps": kpss[i].tolist(), "score": float(scores[i])}
            for i in range(len(scores))
        ]

    def get_embedding(self, img_bgr: np.ndarray, kps) -> np.ndarray:
        """对单张人脸（5 关键点）对齐并提取 512 维 L2 归一化特征。"""
        self.load()
        kps = np.asarray(kps, dtype=np.float32).reshape(5, 2)
        m = self._estimate_affine(kps, ARC_SRC)
        if m is None:
            raise ValueError("人脸对齐失败，关键点无效")
        aimg = cv2.warpAffine(img_bgr, m, (112, 112), borderValue=0.0)
        blob = cv2.dnn.blobFromImage(
            aimg, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True
        )
        input_name = self._rec.get_inputs()[0].name
        emb = self._rec.run(None, {input_name: blob})[0].ravel()
        norm = np.linalg.norm(emb)
        if norm == 0:
            return emb.astype(np.float32)
        return (emb / norm).astype(np.float32)

    @staticmethod
    def _estimate_affine(src: np.ndarray, dst: np.ndarray):
        """估算 5 点相似变换矩阵，带降级策略。"""
        for fn in ("estimateAffineTransform2D", "estimateAffinePartial2D"):
            if hasattr(cv2, fn):
                try:
                    m, _ = getattr(cv2, fn)(src, dst)
                    return m
                except cv2.error:
                    continue
        return None

    def extract_largest_face(self, img_bgr: np.ndarray):
        """检测并返回面积最大的人脸及其 512 维特征；无人脸返回 None。"""
        faces = self.detect(img_bgr)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
        emb = self.get_embedding(img_bgr, face["kps"])
        return {"embedding": emb, "bbox": face["bbox"], "score": face["score"]}

    def liveness_probs(self, img_bgr: np.ndarray, bbox):
        """返回 3 类 softmax [打印攻击, 真人, 翻拍攻击]；活体模型不可用或框非法时返回 None。"""
        self.load()
        if self._live is None:
            return None
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = img_bgr.shape[:2]
        box_w, box_h = x2 - x1, y2 - y1
        if box_w <= 0 or box_h <= 0:
            return None
        # 2.7 倍放大裁剪（受图像边界限制）
        scale = min((w - 1) / box_w, (h - 1) / box_h, LIVENESS_SCALE)
        new_w, new_h = box_w * scale, box_h * scale
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        x1n = max(0, int(cx - new_w / 2))
        y1n = max(0, int(cy - new_h / 2))
        x2n = min(w, int(cx + new_w / 2))
        y2n = min(h, int(cy + new_h / 2))
        if x2n - x1n < 8 or y2n - y1n < 8:
            return None
        crop = cv2.resize(img_bgr[y1n:y2n, x1n:x2n], (80, 80))
        # 注意：按 LIVENESS_INPUT_SCALE 标定，勿改回 /255（详见常量处注释）
        norm = crop.astype(np.float32) / 255.0 * LIVENESS_INPUT_SCALE
        blob = norm.transpose(2, 0, 1)[np.newaxis, ...]
        out = self._live.run(None, {self._live.get_inputs()[0].name: blob})[0][0]
        e = np.exp(out - out.max())
        return e / e.sum()

    def check_liveness(self, img_bgr: np.ndarray, bbox) -> float | None:
        """静默活体检测：返回真人概率（0~1）；活体模型不可用时返回 None。"""
        self.load()
        if self._live is None:
            return None
        probs = self.liveness_probs(img_bgr, bbox)
        if probs is None:
            return 0.0  # 框非法 → 视为不可信
        return float(probs[LIVENESS_LIVE_INDEX])  # live 类概率（索引 1）

    def align_face(self, img_bgr: np.ndarray, bbox) -> np.ndarray:
        """把人脸按框中心裁剪/缩放到 192x192 对齐空间。

        这个空间的数值与「人物离镜头远近」无关，因此同一块区域坐标可以跨帧复用 ——
        动作校验正是靠这一点用「固定 ROI」测量，避免 ROI 跟着表情一起缩放。
        只依赖 bbox，不跑模型，开销极低（约 0.1ms）。
        """
        x1, y1, x2, y2 = [float(v) for v in bbox]
        w, h = x2 - x1, y2 - y1
        if w <= 1 or h <= 1:
            raise ValueError("人脸框无效")
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        scale = LANDMARK_SIZE / (max(w, h) * LANDMARK_MARGIN)
        m = cv2.getRotationMatrix2D((cx, cy), 0, scale)
        m[0, 2] += LANDMARK_SIZE / 2 - cx
        m[1, 2] += LANDMARK_SIZE / 2 - cy
        return cv2.warpAffine(img_bgr, m, (LANDMARK_SIZE, LANDMARK_SIZE), borderValue=0.0)

    def landmarks(self, img_bgr: np.ndarray, bbox):
        """提取 106 点关键点，返回 (points[N,2], crop)。

        坐标系说明（重要）：返回的坐标是 **192x192 对齐人脸空间** 中的像素坐标，
        不是原图坐标。因为对齐时按人脸框尺寸做了缩放，所以这个空间的数值天然
        与「人脸离镜头远近」无关 —— 同一块区域坐标可以跨帧复用。

        对齐方式沿用 InsightFace 官方 Landmark 类：
        以人脸框中心为中心、按 MAX(w,h)*1.5 放大、铺满 192x192。
        """
        self.load()
        if self._lmd is None:
            raise RuntimeError("缺少关键点模型 2d106det.onnx，无法进行动作校验")
        crop = self.align_face(img_bgr, bbox)
        blob = crop.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
        out = self._lmd.run(None, {self._lmd.get_inputs()[0].name: blob})[0]
        # 模型输出为 [-1,1] 归一化坐标，映射回 192 空间
        pts = (np.asarray(out, dtype=np.float32).reshape(-1, 2) + 1.0) * (LANDMARK_SIZE // 2)
        return pts, crop

    def verify(self, emb1: np.ndarray, emb2: np.ndarray, threshold: float = 0.45) -> bool:
        """判断两个特征是否为同一人（余弦相似度超过阈值）。"""
        return cosine_similarity(emb1, emb2) >= threshold


# 全局单例
recognizer = Recognizer()
