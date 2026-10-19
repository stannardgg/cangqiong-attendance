"""苍穹签到系统 — 「最近识别记录」内存缓存 + 定时自动清理。

为什么单独一个模块
------------------
这套系统里跟人脸有关的数据其实分三类，寿命完全不同，必须区别对待：

  1. 注册人脸的特征向量（faces 表）—— **长期资产**，删了学生就签不了到，绝不能自动清。
  2. 签到流水（attendance 表）—— **教学证据**，同样不能自动清。
  3. 「刚才识别到了谁」这类临时信息 —— 用完即弃，留着只是隐私负担。

本模块只负责第 3 类；同时负责清掉 data/debug/ 里可能残留的调试画面
—— 那是全项目**唯一**会把「人脸图像」写到磁盘的地方（且默认关闭）。

三条硬约束
----------
· **只存内存，不落盘。** 进程一退出内容就没了，磁盘上不留任何残迹。
· **只记结果**（姓名 / 学号 / 时间 / 相似度），不记画面、不记特征向量。
· **每条最多存活 TTL**（默认 2 小时），到期自动消失，不需要人来点。

定时机制
--------
后台守护线程每 SWEEP_INTERVAL（默认 60 秒）醒一次：
  · 删掉超过 TTL 的条目（所以「最近两小时内识别到的人」才看得到）；
  · 每满 DEBUG_CLEAN_INTERVAL（默认 2 小时）顺带清空 data/debug/。

用 ``threading.Event.wait`` 而不是 ``time.sleep``：停止时能立刻把线程唤醒，
不会让进程关闭白白多等 60 秒。
"""
import os
import threading
import time
from datetime import datetime

# ---- 可调参数 ----
# 都留了环境变量出口：演示时可以临时调小（例如 TTL=10 秒）当场看到自动清理的效果。
TTL_SECONDS = int(os.environ.get("CANGQIONG_RECENT_TTL_SECONDS", "7200"))                        # 2 小时
SWEEP_INTERVAL = int(os.environ.get("CANGQIONG_SWEEP_INTERVAL_SECONDS", "60"))                   # 每 60 秒扫一次
DEBUG_CLEAN_INTERVAL = int(os.environ.get("CANGQIONG_DEBUG_CLEAN_INTERVAL_SECONDS", "7200"))     # 每 2 小时清调试帧
MAX_ENTRIES = int(os.environ.get("CANGQIONG_RECENT_MAX_ENTRIES", "500"))                         # 内存条数上限

_lock = threading.Lock()
# 键用 (course_id, student_id)：同一个人在同一个课程里只占一行，重复识别只刷新时间与次数。
_entries: dict[tuple[int, int], dict] = {}

_stop = threading.Event()
_thread: threading.Thread | None = None
_debug_dir: str | None = None
_next_debug_clean = 0.0
_started_at = ""
_last_sweep_at = ""
_last_debug_clean_at = ""
_expired_total = 0   # 累计被自动清理掉的条数（用来证明「机制真的在跑」）
_cleared_total = 0   # 累计被手动清空的条数
_debug_removed_total = 0


def _now_str() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ---- 写入 ----

def record(*, course_id: int, course_name: str, student_id: int, name: str,
           student_no: str, similarity: float) -> None:
    """记一次成功识别。同一个人同一课程会合并成一行，只刷新时间与次数。"""
    key = (int(course_id), int(student_id))
    now_mono = time.monotonic()
    now_str = _now_str()
    with _lock:
        entry = _entries.get(key)
        if entry is None:
            _entries[key] = {
                "course_id": int(course_id),
                "course_name": course_name or "",
                "student_id": int(student_id),
                "name": name or "",
                "student_no": student_no or "",
                "similarity": round(float(similarity), 4),
                "times": 1,
                "first_at": now_str,
                "last_at": now_str,
                "_first_mono": now_mono,
                "_last_mono": now_mono,   # 过期判断只看这一项
            }
        else:
            entry["times"] += 1
            entry["similarity"] = round(float(similarity), 4)
            entry["last_at"] = now_str
            entry["_last_mono"] = now_mono
            if course_name:
                entry["course_name"] = course_name
            if name:
                entry["name"] = name
            if student_no:
                entry["student_no"] = student_no
        _evict_locked()


def _evict_locked() -> int:
    """超过条数上限时，丢掉最久没被识别到的那几条，防止长时间上课把内存撑满。"""
    removed = 0
    while len(_entries) > MAX_ENTRIES:
        oldest = min(_entries, key=lambda k: _entries[k]["_last_mono"])
        _entries.pop(oldest, None)
        removed += 1
    return removed


# ---- 读取 / 手动清空 ----

def snapshot(course_id: int | None = None) -> dict:
    """返回缓存内容（最近识别到的排前面）与清理机制的运行状态。"""
    now_mono = time.monotonic()
    with _lock:
        raw = []
        for entry in _entries.values():
            if course_id is not None and entry["course_id"] != int(course_id):
                continue
            raw.append((entry["_last_mono"], {
                "course_id": entry["course_id"],
                "course_name": entry["course_name"],
                "student_id": entry["student_id"],
                "name": entry["name"],
                "student_no": entry["student_no"],
                "similarity": entry["similarity"],
                "times": entry["times"],
                "first_at": entry["first_at"],
                "last_at": entry["last_at"],
                # 还差多少秒被自动清掉（前端直接显示，不用自己算）
                "expires_in": max(0, int(TTL_SECONDS - (now_mono - entry["_last_mono"]))),
            }))
        raw.sort(key=lambda pair: pair[0], reverse=True)   # 最近识别的排前面
        items = [item for _, item in raw]
        total = len(_entries)
        expired_total = _expired_total
        cleared_total = _cleared_total
        debug_removed_total = _debug_removed_total

    return {
        "items": items,
        "count": len(items),            # 本次筛选出的条数
        "total": total,                 # 缓存总条数
        "ttl_seconds": TTL_SECONDS,
        "sweep_interval_seconds": SWEEP_INTERVAL,
        "debug_clean_interval_seconds": DEBUG_CLEAN_INTERVAL,
        "max_entries": MAX_ENTRIES,
        "started_at": _started_at,
        "last_sweep_at": _last_sweep_at,
        "last_debug_clean_at": _last_debug_clean_at,
        "debug_dir": _debug_dir,
        "auto_clean_running": bool(_thread is not None and _thread.is_alive()),
        "expired_total": expired_total,          # 已被自动清掉多少条
        "cleared_total": cleared_total,          # 已被手动清掉多少条
        "debug_removed_total": debug_removed_total,
    }


def clear() -> int:
    """立刻清空缓存（页面上点「立即清空」）。返回清掉的条数。"""
    global _cleared_total
    with _lock:
        n = len(_entries)
        _entries.clear()
        _cleared_total += n
    return n


# ---- 自动清理 ----

def sweep() -> dict:
    """执行一次清理：删掉过期条目；到期时顺带清空调试画面目录。"""
    global _last_sweep_at, _next_debug_clean, _last_debug_clean_at, _expired_total, _debug_removed_total
    now_mono = time.monotonic()

    with _lock:
        # ① 过期条目：按「最后一次被识别」算起，超过 TTL 就丢
        stale = [k for k, e in _entries.items() if now_mono - e["_last_mono"] >= TTL_SECONDS]
        for k in stale:
            _entries.pop(k, None)
        _expired_total += len(stale)
        _last_sweep_at = _now_str()
        # ② 是否轮到清调试画面（IO 放到锁外做，别占着锁）
        due_debug = _debug_dir is not None and now_mono >= _next_debug_clean
        if due_debug:
            _next_debug_clean = now_mono + DEBUG_CLEAN_INTERVAL

    debug_removed = 0
    if due_debug:
        debug_removed = _clean_debug_dir()
        with _lock:
            _debug_removed_total += debug_removed
            _last_debug_clean_at = _now_str()

    return {"expired": len(stale), "debug_files_removed": debug_removed}


def _clean_debug_dir() -> int:
    """清空调试画面目录。只删文件、不递归删目录，避免误伤。返回删除的文件数。"""
    if not _debug_dir or not os.path.isdir(_debug_dir):
        return 0
    removed = 0
    try:
        for filename in os.listdir(_debug_dir):
            path = os.path.join(_debug_dir, filename)
            if not os.path.isfile(path):
                continue
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
    except OSError:
        pass
    return removed


def _loop() -> None:
    """后台守护线程主体。任何异常都吞掉 —— 清理线程绝不能反过来影响签到服务。"""
    while not _stop.wait(SWEEP_INTERVAL):
        try:
            sweep()
        except Exception:
            pass


def start_auto_clean(debug_dir: str) -> None:
    """启动后台清理线程，并顺手清掉上一次运行遗留的调试画面。"""
    global _thread, _debug_dir, _next_debug_clean, _started_at, _last_debug_clean_at, _debug_removed_total
    _debug_dir = debug_dir
    _next_debug_clean = time.monotonic() + DEBUG_CLEAN_INTERVAL
    _started_at = _now_str()

    leftover = _clean_debug_dir()   # 启动时就清一次：上次跑崩留下的画面不该留到这次
    if leftover:
        _debug_removed_total += leftover
        _last_debug_clean_at = _now_str()

    if _thread is not None and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(target=_loop, name="recent-cache-cleaner", daemon=True)
    _thread.start()


def stop_auto_clean() -> None:
    """停止后台线程（服务关闭时调用）。"""
    _stop.set()
    thread = _thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=2.0)
