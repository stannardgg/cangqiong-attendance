"""苍穹签到系统 — FastAPI 后端入口。"""
import base64
import json
import os
import sqlite3
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles

from . import challenge as ch
from . import database as db
from . import liveness
from . import recent_cache
from .models import (
    ChallengeRequest,
    Course,
    CourseCreate,
    FaceRegister,
    ManualAttendance,
    ProbeRequest,
    Student,
    StudentCreate,
    VerifyRequest,
)
from .recognizer import (
    LIVENESS_THRESHOLD,
    cosine_similarity,
    recognizer,
)

RECOG_THRESHOLD = 0.5  # 签到识别阈值（余弦相似度，ArcFace 特征）
MAX_FRAMES = 30        # 单次校验允许的最大帧数（防超大请求体）

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# ---- 人脸写权限：只允许「在运行本系统的这台电脑上」操作 ----
# 需求：手机可以签到，但不能上传 / 修改人脸。
# 这里不是靠前端藏按钮，而是后端真的挡 —— 否则直接调接口就绕过去了。
#
# 判断依据（两条都要满足）：
#   ① 来源地址是回环地址（127.0.0.1 / ::1）→ 排除手机走局域网访问的情况；
#   ② 请求里没有反向代理头 → 排除手机走公网隧道的情况
#      （隧道客户端在本机运行，会把请求转发到 localhost，来源看着像本机，
#        但一定会带上 X-Forwarded-For / CF-Connecting-IP 之类的头）。
PROXY_HEADERS = (
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-proto",
    "x-real-ip",
    "cf-connecting-ip",
    "forwarded",
)
LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")


def is_local_request(request: Request) -> bool:
    """请求是否来自「在运行本系统的这台电脑上」。

    两条都要满足：
      ① 来源是回环地址（127.0.0.1 / ::1）→ 排除手机走局域网；
      ② 不带反向代理头 → 排除手机走公网隧道
        （隧道客户端在本机运行，转发到 localhost 时来源看着像本机，
          但一定会带上 X-Forwarded-For / CF-Connecting-IP 之类的头）。
    """
    if any(h in request.headers for h in PROXY_HEADERS):
        return False
    host = request.client.host if request.client else ""
    return host in LOOPBACK_HOSTS


def can_edit_faces(request: Request) -> bool:
    """当前这次请求是否允许写人脸数据（注册 / 删除）。"""
    return is_local_request(request)


def require_face_write_permission(request: Request) -> None:
    """人脸写操作的守卫，不允许则抛 403。"""
    if not can_edit_faces(request):
        raise HTTPException(
            403,
            "人脸数据只能在这台运行签到系统的电脑上录入或删除。"
            "手机端仅支持签到，如需修改人脸请到电脑上操作。",
        )


def require_local_view(request: Request, what: str) -> None:
    """本机专属操作的守卫（查看名单、清空记录等），不允许则抛 403。

    和 require_face_write_permission 同一个判断口径：能看名单 = 能在这台电脑上打开。
    界面已经把入口藏掉了，这里再挡一层，保证「直接敲接口地址」也拿不到别人的数据。
    """
    if not is_local_request(request):
        raise HTTPException(
            403,
            f"{what}只能在这台运行签到系统的电脑上操作。"
            "手机端仅支持刷脸签到，看不到其他学生的信息。",
        )

# 诊断开关：设置环境变量 CANGQIONG_DEBUG_FRAMES=1 时，把签到帧与判定结果落盘到 data/debug/
# 注意：这是全项目**唯一**会把「人脸图像」写到磁盘的地方，且默认关闭。
# 万一开了忘了关，recent_cache 的后台线程每 2 小时会把这个目录清空一次，启动时也会先清一遍。
DEBUG_DIR = os.path.join(BASE_DIR, "data", "debug")
DEBUG_FRAMES = os.environ.get("CANGQIONG_DEBUG_FRAMES") == "1"
_last_debug_ts = 0.0


def _save_debug_frame(img, frames) -> None:
    """诊断用：把最近一帧与其判定结果写盘（节流 3 秒），便于离线复现识别问题。"""
    global _last_debug_ts
    if not DEBUG_FRAMES:
        return
    now = time.time()
    if now - _last_debug_ts < 3.0:
        return
    _last_debug_ts = now
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        # 注意：cv2.imwrite 不支持中文路径，统一用 imencode + 二进制写入
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if ok:
            with open(os.path.join(DEBUG_DIR, "last_frame.jpg"), "wb") as f:
                f.write(buf.tobytes())
        with open(os.path.join(DEBUG_DIR, "last_frame_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"frames": frames}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def decode_image(data: str) -> np.ndarray:
    """把 base64（可含 data:image 前缀）解码为 BGR 图像。"""
    if data.startswith("data:") and "," in data:
        data = data.split(",", 1)[1]
    raw = base64.b64decode(data)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "图片解码失败")
    return img


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    try:
        recognizer.load()  # 启动即预加载模型（约 0.3s），健康检查可见「就绪」
    except Exception:
        pass  # 模型缺失不阻塞启动，识别时会报明确错误
    # 启动「最近识别记录」的自动清理：每 60 秒扫掉过期条目，每 2 小时清一次调试画面
    recent_cache.start_auto_clean(DEBUG_DIR)
    try:
        yield
    finally:
        recent_cache.stop_auto_clean()


app = FastAPI(title="苍穹签到系统", version="0.1.0", lifespan=lifespan)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "苍穹签到系统",
        "version": "0.2.0",
        "model_ready": recognizer.is_ready(),
        "liveness_ready": recognizer.has_liveness(),
        "landmark_ready": recognizer.has_landmark(),
    }


@app.get("/api/capabilities")
def capabilities(request: Request):
    """告诉前端「当前这台设备能做什么」，前端据此裁剪界面。

    · is_local_device = 在运行本系统的电脑上打开 → 完整功能
    · 手机（局域网或公网隧道）→ 只剩「实时签到」一个页面，
      而它连名单都看不到：只有刷脸成功后，才会在卡片上看到本人的姓名与学号。
      学生/课程管理、人脸录入、签到记录全部隐藏，且人脸写操作与签到记录查询
      在后端也会被 403 拦掉。
    """
    local = is_local_request(request)
    return {
        "is_local_device": local,
        "can_edit_faces": local,
        "can_attendance": True,          # 手机也允许签到
        "can_view_roster": local,        # 手机不展示「谁到了 / 谁没到」的名单
        "can_manage": local,             # 学生 / 课程的增删改查
        "can_view_records": local,       # 签到记录里是全班名单，手机端不给看
        "can_clear_recent": local,       # 「最近识别记录」缓存含全班姓名，同样只给本机
        "note": "人脸录入、名单与签到记录仅限运行本系统的电脑",
    }


# ---- 学生 CRUD ----

@app.get("/api/students", response_model=list[Student])
def list_students(class_name: str | None = Query(None, description="按班级筛选")):
    return db.list_students(class_name)


@app.post("/api/students", response_model=Student, status_code=201)
def create_student(payload: StudentCreate):
    try:
        return db.create_student(payload.name, payload.student_no, payload.class_name)
    except sqlite3.IntegrityError:
        raise HTTPException(409, "学号已存在")


@app.get("/api/students/{student_id}", response_model=Student)
def get_student(student_id: int):
    s = db.get_student(student_id)
    if s is None:
        raise HTTPException(404, "学生不存在")
    return s


@app.put("/api/students/{student_id}", response_model=Student)
def update_student(student_id: int, payload: StudentCreate):
    if db.get_student(student_id) is None:
        raise HTTPException(404, "学生不存在")
    try:
        db.update_student(student_id, payload.name, payload.student_no, payload.class_name)
    except sqlite3.IntegrityError:
        raise HTTPException(409, "学号已存在")
    return db.get_student(student_id)


@app.delete("/api/students/{student_id}")
def delete_student(student_id: int, request: Request):
    # 删除学生会连带删除其人脸特征，因此同样纳入「仅限本机」保护，
    # 否则手机端可以靠删学生绕过人脸不可改的限制。
    require_face_write_permission(request)
    if not db.delete_student(student_id):
        raise HTTPException(404, "学生不存在")
    return {"ok": True}


# ---- 课程 CRUD ----

@app.get("/api/courses", response_model=list[Course])
def list_courses():
    return db.list_courses()


@app.post("/api/courses", response_model=Course, status_code=201)
def create_course(payload: CourseCreate):
    return db.create_course(payload.name, payload.class_name)


@app.get("/api/courses/{course_id}", response_model=Course)
def get_course(course_id: int):
    c = db.get_course(course_id)
    if c is None:
        raise HTTPException(404, "课程不存在")
    return c


@app.put("/api/courses/{course_id}", response_model=Course)
def update_course(course_id: int, payload: CourseCreate):
    if db.get_course(course_id) is None:
        raise HTTPException(404, "课程不存在")
    db.update_course(course_id, payload.name, payload.class_name)
    return db.get_course(course_id)


@app.delete("/api/courses/{course_id}")
def delete_course(course_id: int):
    if not db.delete_course(course_id):
        raise HTTPException(404, "课程不存在")
    return {"ok": True}


# ---- 人脸注册 ----
@app.post("/api/students/{student_id}/faces")
def register_face(student_id: int, payload: FaceRegister, request: Request):
    """为指定学生注册一张人脸（自动检测最大人脸并提取特征存库）。"""
    require_face_write_permission(request)
    if db.get_student(student_id) is None:
        raise HTTPException(404, "学生不存在")
    img = decode_image(payload.image)
    result = recognizer.extract_largest_face(img)
    if result is None:
        raise HTTPException(400, "未检测到人脸，请调整光线与角度后重试")
    face_id = db.add_face(student_id, result["embedding"])
    return {"ok": True, "face_id": face_id, "faces_total": db.count_faces(student_id)}


@app.get("/api/students/{student_id}/faces")
def student_faces(student_id: int):
    """查看某学生已注册的人脸特征数量。"""
    if db.get_student(student_id) is None:
        raise HTTPException(404, "学生不存在")
    return {"student_id": student_id, "faces_total": db.count_faces(student_id)}


@app.delete("/api/students/{student_id}/faces")
def delete_student_faces(student_id: int, request: Request):
    """删除某学生的全部人脸特征（仅限本机电脑操作）。"""
    require_face_write_permission(request)
    if db.get_student(student_id) is None:
        raise HTTPException(404, "学生不存在")
    return {"ok": True, "deleted": db.delete_faces(student_id)}


# ---- 实时签到（时序活体 + 动作挑战，防照片/视频代签）----
#
# 与旧版的关键差别：旧版每帧独立提交，后端没有任何时序概念，单帧活体模型
# 对静态照片几乎必过（实测 lena.jpg 真人概率 0.9962）。现在的链路是：
#
#   ① probe   ：单帧探测，只在「画面里出现一张够大的人脸」时唤醒前端，绝不写签到
#   ② challenge：服务端随机指定动作（眨眼 / 张嘴），签发 45 秒一次性令牌
#   ③ verify  ：提交 1.4 秒内的连续帧，后端做时序活体 + 动作判定 + 1:N 比对，全部通过才写签到
#
# 三个接口都不写库（除 verify 最后一步），也都是手机端可用的 —— 学生就是用手机签到的。

def _largest(faces: list[dict]) -> dict | None:
    if not faces:
        return None
    return max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))


@app.post("/api/attendance/probe")
def probe(payload: ProbeRequest):
    """单帧探测：回答「画面里现在能不能开始验证」。不写任何签到记录。"""
    if db.get_course(payload.course_id) is None:
        raise HTTPException(404, "课程不存在")
    img = decode_image(payload.image)
    faces = recognizer.detect(img)
    big = [f for f in faces if (f["bbox"][2] - f["bbox"][0]) >= liveness.MIN_FACE_PX]
    if len(faces) > 1 and len(big) > 1:
        return {"ready": False, "faces": len(faces), "hint": "画面中只能有一个人，请让其他人离开镜头"}
    face = _largest(faces)
    if face is None:
        return {"ready": False, "faces": 0, "hint": ""}
    width = float(face["bbox"][2] - face["bbox"][0])
    live = recognizer.check_liveness(img, face["bbox"])
    ready = width >= liveness.MIN_FACE_PX and (live is None or live >= LIVENESS_THRESHOLD)
    hint = "" if ready else ("请靠近摄像头一些" if width < liveness.MIN_FACE_PX else "")
    return {
        "ready": ready,
        "faces": len(faces),
        "face_width_px": round(width, 1),
        "liveness": None if live is None else round(float(live), 4),
        "hint": hint,
    }


@app.post("/api/attendance/challenge")
def issue_challenge(payload: ChallengeRequest):
    """签发一次性动作挑战。动作由服务端随机决定，客户端无法预知。"""
    if db.get_course(payload.course_id) is None:
        raise HTTPException(404, "课程不存在")
    if not liveness.models_ready():
        raise HTTPException(503, "缺少活体或关键点模型，无法进行防代签校验")
    return ch.issue(payload.course_id)


@app.post("/api/attendance/verify")
def verify(payload: VerifyRequest):
    """校验一段连续帧并完成签到。

    注意：动作不通过时返回 HTTP 200 + ok=false + reason，而不是 4xx ——
    这是正常的业务结果（学生没做动作），前端据此提示重试，不该当成接口故障。
    """
    course = db.get_course(payload.course_id)
    if course is None:
        raise HTTPException(404, "课程不存在")
    if len(payload.frames) > MAX_FRAMES:
        raise HTTPException(400, f"帧数过多（上限 {MAX_FRAMES}），请缩短采集序列")

    chall, err = ch.consume(payload.challenge_id, payload.course_id)
    if err is not None:
        msg = {
            "not_found": "验证请求已失效，请重新开始",
            "expired": "验证超时（超过 45 秒），请重新开始",
            "used": "这次验证已经用过了，请重新开始",
            "course_mismatch": "验证与课程不匹配，请重新开始",
        }.get(err, "验证失败，请重新开始")
        return {"ok": False, "reason": err, "detail": msg, "hint": msg}

    frames = [decode_image(f) for f in payload.frames]
    ref_img = decode_image(payload.ref) if payload.ref else None
    result = liveness.analyze_sequence(frames, chall.action, ref=ref_img)
    metrics = result.get("metrics", {})
    _save_debug_frame(frames[len(frames) // 2], [{
        "action": chall.action,
        "ok": result["ok"],
        "reason": result.get("reason"),
        "metrics": metrics,
    }])

    if not result["ok"]:
        return {
            "ok": False,
            "reason": result["reason"],
            "detail": result["detail"],
            "hint": result["detail"],
            "action": chall.action,
            "metrics": metrics,
            "spoof": result["reason"] == "spoof",
        }

    all_embs = db.list_all_face_embeddings()
    if not all_embs:
        return {"ok": False, "reason": "no_faces_registered",
                "detail": "尚未注册任何学生人脸", "hint": "请先在本机为学生录入人脸"}

    emb = result["embedding"]
    best_sid, best_sim = -1, -1.0
    for sid, ref in all_embs:
        sim = cosine_similarity(emb, ref)
        if sim > best_sim:
            best_sim, best_sid = sim, sid
    best_sim = round(float(best_sim), 4)
    if best_sim < RECOG_THRESHOLD:
        return {"ok": False, "reason": "no_match", "best_similarity": best_sim,
                "detail": "未识别出已注册的学生", "hint": "请确认本条人脸已录入，或换一个角度重试"}

    student = db.get_student(best_sid)
    recent_cache.record(
        course_id=payload.course_id,
        course_name=course["name"],
        student_id=best_sid,
        name=student["name"],
        student_no=student["student_no"],
        similarity=best_sim,
    )
    is_new = db.mark_attendance(payload.course_id, best_sid)
    return {
        "ok": True,
        "action": chall.action,
        "result": "attended" if is_new else "already",
        "student_id": best_sid,
        "name": student["name"],
        "student_no": student["student_no"],
        "similarity": best_sim,
        "liveness": result.get("liveness"),
        "check_in_at": db.now_str() if is_new else "",
        "metrics": metrics,
    }


@app.post("/api/attendance/manual")
def manual_attendance(payload: ManualAttendance, request: Request):
    """手工补签：动作校验连续失败（镜头脏、逆光、戴口罩等）时的兜底通道。

    设计取舍：宁可留一条「老师看得见、留得下痕迹」的人工通道，
    也不要把学生堵在门外逼他去想办法「过验证」——后者反而会诱发代签。
    记录 status='manual'，签到记录里会显示成「手工补签」，与刷脸出勤可区分。
    仅限本机电脑操作，不存在远程补签。
    """
    require_local_view(request, "手工补签")
    if db.get_course(payload.course_id) is None:
        raise HTTPException(404, "课程不存在")
    student = db.get_student(payload.student_id)
    if student is None:
        raise HTTPException(404, "学生不存在")
    is_new = db.mark_attendance(payload.course_id, payload.student_id, status="manual")
    return {
        "ok": True,
        "result": "attended" if is_new else "already",
        "student_id": payload.student_id,
        "name": student["name"],
        "student_no": student["student_no"],
        "check_in_at": db.now_str() if is_new else "",
    }


@app.get("/api/attendance/diagnostics")
def attendance_diagnostics(request: Request):
    """最近一次序列校验的指标（不含图像与特征向量），仅限本机排查使用。"""
    require_local_view(request, "查看活体诊断信息")
    return {"last": liveness.last_report(), "challenges": ch.stats(),
            "models": {"liveness": recognizer.has_liveness(), "landmark": recognizer.has_landmark()}}


@app.get("/api/attendance")
def attendance_list(request: Request, course_id: int, date: str | None = None):
    """查询某课程某日期的出勤记录。返回的是全班名单，仅限本机查看。"""
    require_local_view(request, "查看签到记录")
    if db.get_course(course_id) is None:
        raise HTTPException(404, "课程不存在")
    return db.list_attendance(course_id, date)


@app.delete("/api/attendance")
def clear_attendance(request: Request, course_id: int, date: str | None = None):
    """清空某课程的签到记录（指定 date 则只清该日），用于重新签到。仅限本机操作。"""
    require_local_view(request, "清空签到记录")
    if db.get_course(course_id) is None:
        raise HTTPException(404, "课程不存在")
    return {"ok": True, "deleted": db.clear_attendance(course_id, date)}


# ---- 最近识别记录的临时缓存（纯内存，与上面的签到流水完全是两回事）----
# 说明：签到流水存在数据库里、是长期证据，不会自动清。
# 这里缓存的是「刚才这两个小时里识别到了谁」这种临时信息，只放在内存里，
# 超过 TTL（默认 2 小时）会被后台线程自动清掉；进程一退出也立刻没有。

@app.get("/api/recent")
def recent_list(request: Request, course_id: int | None = None):
    """查看「最近识别记录」缓存及其清理机制的运行状态。仅限本机。"""
    require_local_view(request, "查看最近识别缓存")
    if course_id is not None and db.get_course(course_id) is None:
        raise HTTPException(404, "课程不存在")
    return recent_cache.snapshot(course_id)


@app.delete("/api/recent")
def recent_clear(request: Request):
    """立刻清空「最近识别记录」缓存。仅限本机。"""
    require_local_view(request, "清空最近识别缓存")
    return {"ok": True, "cleared": recent_cache.clear()}


# 静态文件挂载放在 API 路由之后，"/" 作为兜底
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
