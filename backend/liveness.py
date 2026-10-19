"""苍穹签到系统 — 时序活体 + 主动动作校验（防代签核心）。

为什么需要这个模块
------------------
单帧活体（MiniFASNetV2）判断的是「这张图的皮肤纹理像不像真人」，它拿不到
「这一帧是实时拍摄还是屏幕上的画面」这个信息。实测一张静态照片文件
（data/test/real/lena.jpg）的真人概率为 **0.9962**，远高于 0.5 阈值 ——
即单帧模型对高质量照片/视频回放基本无效。

因此必须引入两个单帧拿不到的信息：

    ① 时间维度 —— 同一人在连续多帧中的生理变化（张嘴）；
    ② 随机性   —— 服务端临时下发、事先无法预测的指令。

动作只有「张嘴」，随机性放在**次数**上
--------------------------------------
服务端随机要求「张一次」或「张两次嘴」。这样：
    · 静态照片永远做不到          → 拒
    · 预录的「张一次嘴」视频对不上「张两次」的指令 → 拒
    · 真人无论哪条指令都很容易做到 → 误拒风险接近零

这是被数据逼出来的设计。下面三个候选动作都实测过、都不能用：

| 候选 | 实测结果 | 结论 |
|---|---|---|
| 闭眼 | 一次自然眨眼 100~300ms，而采样约 100ms/帧，最多抓到 1 帧且常是半闭合 | 真人做对也判不过，弃用 |
| 转头 | 关键点回归器有强整体形状先验，**会把偏转归一化掉**：绕竖轴转 20°，眼部指标只变 0.03~0.05；把人脸放大到 717px 画布、绕人脸中心转，±25° 就连检测都失败 | 量不准，弃用 |
| 抬眉 | 眉带暗像素重心随眉毛上移几乎不动（上移 5px，重心只变 0.04，与抖动噪声 0.058 同量级）；「眉带下缘暗占比」在 0.09~0.14 之间无规律波动 | 量不准，弃用 |

教训：**能不能测准，必须先用 `tools/probe_*.py` 量出来再决定**，不能凭直觉选动作。
选一个量不准的动作，代价是学生站在摄像头前反复失败。

设计约束一：动作信号用「区域亮度统计」，不用「关键点开合比」
--------------------------------------------------
实测：嘴部纵向拉伸 35%，关键点开合比只变 1%；把整只眼睛用皮肤色覆盖，
眼部开合比只跌 19%；人脸放大到 432px 仍如此。根因是 2d106det 是
「全连接输出 106 点」的形状回归器，形状先验太强 —— **它擅长定位，不擅长测局部形变**。

改用「嘴区暗像素占比」后，同一个张嘴实验：0.018 → 0.309（17 倍）。
原理直白：闭嘴时嘴区是一块均匀的皮肤/唇，张嘴时出现一个暗腔。
所以：**关键点只用来定位 ROI，判动作用 ROI 内的亮度统计。**

设计约束二：ROI 尺寸必须固定，只允许刚性跟随头部
--------------------------------------------
如果每帧都用自己当帧的关键点外接框当 ROI，嘴张得越开、框越大越走形 ——
等于用会变化的尺子去量正在变化的东西。所以在 192 对齐空间里**只用一帧确定 ROI 尺寸**，
之后所有帧共用同一个尺寸（见 `_roi_box`）。

但也不能完全钉死：相机在动、人在动，画面平移 6 个像素就足以把 ROI 挪到嘴外，
而「ROI 挪出目标」在亮度统计上与「动作」几乎同形（实测手持抖动的照片曾骗过判据）。
因此每帧再用**面部轮廓点（0~32）**估计一个「平移 + 尺度」的刚性变换，
把固定尺寸的 ROI 搬到当前帧：轮廓点不会因为张嘴而改变位置，只跟随头部，
所以它天然满足「跟着头动、不跟着表情动」。

设计约束三：基线必须来自「指令出现之前」的那一帧
--------------------------------------------
早期版本用序列自身的分位数当基线。这有个致命缺陷：**动作占满大部分帧时，
基线就被动作本身污染了**。学生看到指令就张嘴、一直保持到采集结束，
「闭嘴基线」实际取到的是张嘴值，算出来的增幅≈0，判为「没做动作」——
人明明做了，系统说他没做。这不是阈值问题，是基线选错了参照物。

现在前端会把**显示指令之前**的一帧（`ref`）一起提交，服务端在那一帧上测基线。
那一帧学生还没看到指令，必然是中性状态。基线从此与「做了多久」无关。
（`ref` 缺省时退回「按最大空隙把序列分成两堆」的老办法，见 `_two_levels`。）

设计约束四：判据看「连续段」，不看「达标帧总数」
--------------------------------------------
真实动作一定落在**连续的一段时间**里；而对齐误差、JPEG 压缩噪声造成的数值波动
是**零散的单帧尖刺**。早期版本统计达标帧总数，结果一张画面平移 6px 的照片
就能凑够帧数骗过判定。现在统计「连续段」——顺带这一招直接给出了「张了几次」。
"""

import os

import numpy as np

from .recognizer import (
    LANDMARK_MODEL,
    LIVENESS_THRESHOLD,
    cosine_similarity,
    recognizer,
)

# ---- 关键点索引 ----------------------------------------------------------
# 已用 tools/probe_landmark.py 在真实照片上画图核对过，不要凭记忆改这些区间。
MOUTH_IDX = list(range(52, 72))   # 上下唇内外轮廓共 20 点
CONTOUR_IDX = list(range(0, 33))  # 面部轮廓 33 点：只随头部移动，不看表情变化

# ---- 可选动作 ------------------------------------------------------------
# repeat = 服务端随机要求做几次。次数本身就是随机性：照片做不出来，
# 一段「张一次嘴」的录像也满足不了「张两次」。
ACTIONS = {
    "mouth": {
        "label": "张嘴一次",
        "repeat": 1,
        "instruction": "请张嘴约 1 秒",
    },
    "mouth2": {
        "label": "张嘴两次",
        "repeat": 2,
        "instruction": "请连着张两次嘴，每次约半秒",
    },
}

# ---- 阈值（支持环境变量覆盖，见 CLAUDE.md）-------------------------------
MIN_FRAMES = 8            # 少于这么多帧不判（信息不足）
MIN_FACE_PX = int(os.environ.get("CANGQIONG_MIN_FACE_PX", "90"))  # 人脸宽度下限
LIVENESS_OK = LIVENESS_THRESHOLD   # 单帧真人概率阈值（0.5）
IDENTITY_SIM = 0.55       # 首末帧特征余弦相似度下限：低于此视为中途换人

# 张嘴：嘴区暗像素占比相对基线（指令前那一帧）涨多少算「张开」。
# 静态照片该值恒等于基线，因此阈值可以设得不高，兼顾真人张嘴幅度不一。
MOUTH_DARK_ABS = float(os.environ.get("CANGQIONG_MOUTH_DARK", "0.05"))
MOUTH_RISE_RATIO = float(os.environ.get("CANGQIONG_MOUTH_RISE", "1.45"))
MOUTH_HOLD_FRAMES = 2     # 一次张嘴至少连续两帧才算「张开了一次」

# 刚性跟踪的尺度钳位：相邻帧的轮廓尺度不应突变，突变多半是关键点抖动，
# 钳住可以防止一次误检把 ROI 放大到整张脸上。
RIGID_SCALE_MIN = 0.80
RIGID_SCALE_MAX = 1.25

# 诊断用：最近一次序列分析的完整指标（只保留一份，供前端/排查读取）
_last_report: dict | None = None


def _roi_box(pts: np.ndarray, idx: list[int], pad: int = 2) -> list[int]:
    """由关键点算出 ROI 矩形（192 对齐空间坐标）。"""
    p = pts[idx]
    return [
        int(p[:, 0].min()) - pad,
        int(p[:, 1].min()) - pad,
        int(p[:, 0].max()) + pad,
        int(p[:, 1].max()) + pad,
    ]


def _crop(gray: np.ndarray, box: list[int]) -> np.ndarray:
    x1 = max(0, box[0])
    y1 = max(0, box[1])
    x2 = min(gray.shape[1], box[2])
    y2 = min(gray.shape[0], box[3])
    if x2 - x1 < 3 or y2 - y1 < 3:
        return np.empty((0, 0), dtype=np.uint8)
    return gray[y1:y2, x1:x2]


def _darkness(region: np.ndarray) -> float:
    """ROI 内「明显比周围暗」的像素占比（0~1）。

    阈值取 min(90, 均值 - 0.6σ)：既照顾整体偏暗的画面（暗光教室），
    也照顾对比度高的画面（强光直射）。不用固定阈值是为了抗光照差异。
    """
    if region is None or region.size < 12:
        return 0.0
    m = float(region.mean())
    s = float(region.std())
    thresh = min(90.0, m - 0.6 * max(s, 3.0))
    return float((region < thresh).mean())


def _to_gray(img: np.ndarray) -> np.ndarray:
    """BGR → 灰度（整数近似，避免依赖 cv2 的常量命名差异）。"""
    b = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    r = img[:, :, 2].astype(np.float32)
    return (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)


def _runs(flags: list[bool], min_len: int = 1) -> list[tuple[int, int]]:
    """所有长度 ≥ min_len 的连续 True 段，返回 [(起点, 长度), ...]。

    这是「动作发生了没有、发生了几次」的唯一判据来源：
    真动作是连续的一大段，对齐误差/压缩噪声是零散的单帧尖刺。
    """
    out, start, cur = [], 0, 0
    for i, f in enumerate(flags):
        if f:
            if cur == 0:
                start = i
            cur += 1
        else:
            if cur >= min_len:
                out.append((start, cur))
            cur = 0
    if cur >= min_len:
        out.append((start, cur))
    return out


def _longest(segments: list[tuple[int, int]]) -> tuple[int, int]:
    return max(segments, key=lambda s: s[1]) if segments else (0, 0)


def _two_levels(series: list) -> tuple[float, float, float, float]:
    """把一串数值按「最大空隙」分成低/高两堆，返回 (低, 高, 差距, 分界)。

    这是没有参考帧时的兜底基线。动作占满大部分帧时它会失效（两类样本只剩一类），
    正常流程请使用 `ref` 参考帧给的基线。
    """
    xs = sorted(float(v) for v in series)
    if len(xs) < 4:
        return xs[0], xs[-1], xs[-1] - xs[0], (xs[0] + xs[-1]) / 2.0
    gaps = [(xs[i + 1] - xs[i], i) for i in range(len(xs) - 1)]
    _, cut = max(gaps)
    low = float(np.median(xs[:cut + 1]))
    high = float(np.median(xs[cut + 1:]))
    return low, high, high - low, (xs[cut] + xs[cut + 1]) / 2.0


def _rigid_rois(pts_ref: np.ndarray, pts_now: np.ndarray) -> dict:
    """由轮廓点估计「平移 + 尺度」，把参考帧的 ROI 搬到当前帧（不改变尺寸比例）。"""
    c_ref = pts_ref[CONTOUR_IDX]
    c_now = pts_now[CONTOUR_IDX]
    m_ref, m_now = c_ref.mean(0), c_now.mean(0)
    r_ref = float(np.mean(np.linalg.norm(c_ref - m_ref, axis=1)))
    r_now = float(np.mean(np.linalg.norm(c_now - m_now, axis=1)))
    if r_ref < 1e-3 or not np.isfinite(r_now):
        return {"k": 1.0, "center": (float(m_now[0]), float(m_now[1]))}
    k = float(np.clip(r_now / r_ref, RIGID_SCALE_MIN, RIGID_SCALE_MAX))
    return {"k": k, "center": (float(m_now[0]), float(m_now[1])),
            "ref": (float(m_ref[0]), float(m_ref[1]))}


def _map_box(box: list[int], r: dict) -> list[int]:
    """参考帧 ROI → 当前帧 ROI（平移 + 等比例缩放，长宽比保持不变）。"""
    k = r["k"]
    ox, oy = r.get("ref", r["center"])
    cx, cy = r["center"]
    x1, y1, x2, y2 = box
    return [
        int(round((x1 - ox) * k + cx)),
        int(round((y1 - oy) * k + cy)),
        int(round((x2 - ox) * k + cx)),
        int(round((y2 - oy) * k + cy)),
    ]


def _biggest(faces: list) -> dict:
    return max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))


def _fail(reason: str, detail: str, metrics: dict | None = None) -> dict:
    return {"ok": False, "reason": reason, "detail": detail, "metrics": metrics or {}}


def analyze_sequence(frames: list, action: str, ref=None) -> dict:
    """分析一段连续帧，判断是否真人并完成指定动作。

    参数
    ----
    frames : list[np.ndarray]  BGR 图像，按时间顺序，来自同一次采集
    action : str               "mouth" | "mouth2"，来自服务端签发的挑战
    ref    : np.ndarray | None **指令出现之前**拍的一帧，用作基线。
                               缺失时退回 frames[0] 并启用兜底基线算法。

    返回
    ----
    dict，含 ok / reason / detail / metrics / embedding（识别用）。
    """
    global _last_report

    if action not in ACTIONS:
        return _fail("bad_action", "未知的动作指令，请重新发起验证")
    n = len(frames)
    if n < MIN_FRAMES:
        return _fail("not_enough_frames", f"采集帧数不足（{n}/{MIN_FRAMES}），请重试")

    # ---- ① 锚点检测：首 / 中 / 尾各检测一次，其余帧沿用最近的锚点框 ----
    # 只检测 3 次是为了控制耗时（det_10g 在 CPU 上约 105ms/次）。
    # 2.2 秒内的位移很小，沿用框做对齐足够准确。
    anchor_idx = [0, n // 2, n - 1]
    anchors: dict[int, list[dict]] = {i: recognizer.detect(frames[i]) for i in anchor_idx}

    for i in anchor_idx:
        big = [f for f in anchors[i] if (f["bbox"][2] - f["bbox"][0]) >= MIN_FACE_PX]
        if len(big) > 1:
            return _fail("multiple_faces", "画面中只能有一个人，请让其他人离开镜头后重试")
    if not anchors[0]:
        return _fail("no_face", "没有检测到人脸，请正对摄像头后重试")

    main0 = _biggest(anchors[0])
    face_w = float(main0["bbox"][2] - main0["bbox"][0])
    if face_w < MIN_FACE_PX:
        return _fail("too_far", "人脸太小，请靠近摄像头一些，让脸部占画面更大")

    # ---- ② 用「指令前那一帧」定 ROI 与基线 ----
    ref_img = frames[0] if ref is None else ref
    ref_faces = anchors[0] if ref is None else recognizer.detect(ref_img)
    if not ref_faces:
        # 参考帧里没人脸：退回首帧，绝不因此判用户失败
        ref_img, ref_faces = frames[0], anchors[0]
    pts_ref, crop_ref = recognizer.landmarks(ref_img, _biggest(ref_faces)["bbox"])

    roi = _roi_box(pts_ref, MOUTH_IDX)
    base_dark = _darkness(_crop(_to_gray(crop_ref), roi))

    # ---- ③ 逐帧：对齐 → 刚性搬 ROI → 取嘴区暗度 ----
    primary = {i: _biggest(faces) for i, faces in anchors.items() if faces}
    ordered = sorted(primary)
    bbox_for = [primary[min(ordered, key=lambda a: abs(a - i))]["bbox"] for i in range(n)]

    mouth_dark = []
    pts_list = [pts_ref]
    scales = []
    for i, frame in enumerate(frames):
        try:
            pts, crop = recognizer.landmarks(frame, bbox_for[i])
            gray = _to_gray(crop)
        except Exception:
            # 关键点这一帧失败不应该让整次签到失败：退回「不跟踪」。
            pts, gray = pts_list[-1], _to_gray(recognizer.align_face(frame, bbox_for[i]))
        pts_list.append(pts)
        rigid = _rigid_rois(pts_ref, pts)
        scales.append(round(rigid["k"], 3))
        mouth_dark.append(_darkness(_crop(gray, _map_box(roi, rigid))))

    # 关键点漂移：真人会有呼吸/微抖，静态画面几乎不动（软指标，只做展示）
    try:
        drift = float(np.mean(np.linalg.norm(pts_list[-1] - pts_ref, axis=1)))
    except Exception:
        drift = 0.0

    # ---- ④ 活体分数（抽样 5 帧取中位数，单帧偶然值不决定结果）----
    sample = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
    live_scores = []
    for i in sample:
        p = recognizer.check_liveness(frames[i], bbox_for[i])
        if p is not None:
            live_scores.append(float(p))
    live_median = float(np.median(live_scores)) if live_scores else None

    # ---- ⑤ 身份一致性：首帧 vs 尾帧（末帧检测结果直接复用锚点，省一次推理）----
    emb0 = recognizer.get_embedding(frames[0], main0["kps"])
    last_faces = anchors.get(n - 1) or [main0]
    emb1 = recognizer.get_embedding(frames[-1], _biggest(last_faces)["kps"])
    identity_sim = float(cosine_similarity(emb0, emb1))

    metrics = {
        "frames": n,
        "face_width_px": round(face_w, 1),
        "liveness_median": None if live_median is None else round(live_median, 4),
        "liveness_samples": [round(v, 4) for v in live_scores],
        "identity_similarity": round(identity_sim, 4),
        "landmark_drift": round(drift, 3),
        "mouth_dark": [round(v, 3) for v in mouth_dark],
        "roi_mouth": roi,
        "ref_base": {"mouth_dark": round(base_dark, 3), "from_ref_frame": ref is not None},
        "rigid_scale": scales,
    }

    # ---- ⑥ 硬门禁 ----
    if live_median is not None and live_median < LIVENESS_OK:
        _last_report = {"action": action, "ok": False, "reason": "spoof", **metrics}
        return _fail("spoof", "疑似照片或屏幕画面，请本人面对摄像头", metrics)

    if identity_sim < IDENTITY_SIM:
        _last_report = {"action": action, "ok": False, "reason": "identity_switch", **metrics}
        return _fail("identity_switch", "画面中的人前后不一致，请单人正对摄像头重试", metrics)

    action_ok, action_detail = _check_action(action, mouth_dark, base_dark,
                                             from_ref=ref is not None)
    metrics["action_detail"] = action_detail
    _last_report = {"action": action, "ok": action_ok, **metrics}

    if not action_ok:
        return _fail("action_missing", action_detail["hint"], metrics)

    return {
        "ok": True,
        "reason": None,
        "detail": "验证通过",
        "metrics": metrics,
        "embedding": emb0,
        "liveness": live_median,
    }


def _check_action(action: str, mouth: list, base_dark: float, from_ref: bool) -> tuple[bool, dict]:
    """动作是否真的发生、发生了几次。全部与「指令出现前那一帧」比较。"""
    spec = ACTIONS[action]
    need = int(spec["repeat"])

    low, high, gap, _thr = _two_levels(mouth)
    # 基线取「参考帧」与「序列最低值」中较小的那个，两个方向都能兜住：
    #   · 参考帧恰好拍在张嘴中间（基线偏高）→ 序列最低值兜住
    #   · 学生看到指令就一直张着不放（序列最低值偏高）→ 参考帧兜住
    # 照片两种都等于闭嘴值，因此不会因此放松对照片的判定。
    base = min(base_dark, low) if from_ref else low
    flags = [(v - base) >= MOUTH_DARK_ABS * 0.6 for v in mouth]
    segments = _runs(flags, MOUTH_HOLD_FRAMES)
    count = len(segments)
    start, length = _longest(segments)
    seg = mouth[start:start + length] if length else []
    delta = (float(np.median(seg)) - base) if seg else 0.0
    peak = float(np.max(mouth))
    rise = (peak / base) if base > 1e-3 else 0.0

    ok_open = delta >= MOUTH_DARK_ABS and (rise >= MOUTH_RISE_RATIO or base < 0.02)
    ok = ok_open and count >= need

    if not ok_open:
        hint = (f"没有检测到张嘴（暗区增幅 {max(delta, peak - base):.2f}，需 ≥{MOUTH_DARK_ABS:.2f}）。"
                "请把嘴张大一些并保持约 1 秒")
    elif count < need:
        hint = f"只检测到张嘴 {count} 次，指令要求 {need} 次。请先闭上嘴，再连续做 {need} 次"
    else:
        hint = ""

    return ok, {
        "base": round(base, 3),
        "peak": round(peak, 3),
        "rise": round(rise, 2),
        "delta": round(delta, 3),
        "count": count,
        "need_count": need,
        "held_frames": length,
        "need_delta": MOUTH_DARK_ABS,
        "base_from_ref": bool(from_ref),
        "hint": hint,
    }


def last_report() -> dict | None:
    """最近一次序列分析的指标快照，供状态栏与排查使用（不含图像与特征向量）。"""
    return _last_report


def models_ready() -> bool:
    """动作校验依赖的关键点模型是否就位。"""
    return recognizer.has_liveness() and os.path.exists(LANDMARK_MODEL)
