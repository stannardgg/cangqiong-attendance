"""苍穹签到系统 — 一次性动作挑战令牌。

作用（防代签的第二道锁）
------------------------
时序活体只能证明「画面里有人在动」，但一段**预先录好的视频**同样有人在动。
真正的判据是「画面里的人能不能响应一个**当场才知道**的指令」，所以：

    · 动作由**服务端**用 secrets 随机决定，客户端无法预测、无法预演；
    · 令牌 **45 秒**后失效 —— 来不及现录一段；
    · 令牌**只能用一次** —— 同一段序列不能重复提交给多个人用。

存储：纯内存字典，进程退出即消失。签到数据不依赖它，丢了只是重来一次。
"""

import secrets
import threading
import time

from .liveness import ACTIONS

TTL_SECONDS = 45.0        # 令牌有效期
MAX_ENTRIES = 500         # 内存上限，防止异常调用撑爆


class Challenge:
    __slots__ = ("id", "action", "course_id", "created_at", "used_at")

    def __init__(self, challenge_id: str, action: str, course_id: int) -> None:
        self.id = challenge_id
        self.action = action
        self.course_id = course_id
        self.created_at = time.time()
        self.used_at: float | None = None

    @property
    def expired(self) -> bool:
        return (time.time() - self.created_at) > TTL_SECONDS


_lock = threading.Lock()
_store: dict[str, Challenge] = {}


def _purge_locked() -> None:
    """清掉过期/已用条目。调用方必须已持锁。"""
    now = time.time()
    dead = [
        k for k, c in _store.items()
        if (now - c.created_at) > TTL_SECONDS or c.used_at is not None
    ]
    for k in dead:
        _store.pop(k, None)
    # 仍然过多（异常调用）→ 丢弃最早的
    if len(_store) > MAX_ENTRIES:
        for k in sorted(_store, key=lambda k: _store[k].created_at)[: len(_store) - MAX_ENTRIES]:
            _store.pop(k, None)


def issue(course_id: int) -> dict:
    """签发一个新挑战：随机动作 + 一次性令牌。"""
    action = secrets.choice(list(ACTIONS))
    challenge_id = secrets.token_urlsafe(18)
    with _lock:
        _purge_locked()
        _store[challenge_id] = Challenge(challenge_id, action, course_id)
    return {
        "challenge_id": challenge_id,
        "action": action,
        "label": ACTIONS[action]["label"],
        "instruction": ACTIONS[action]["instruction"],
        "expires_in": TTL_SECONDS,
    }


def consume(challenge_id: str, course_id: int) -> tuple[Challenge | None, str | None]:
    """取出并作废令牌。返回 (挑战, 错误码)。

    错误码：not_found / expired / used / course_mismatch
    """
    with _lock:
        c = _store.get(challenge_id)
        if c is None:
            return None, "not_found"
        if c.course_id != course_id:
            return None, "course_mismatch"
        if c.expired:
            _store.pop(challenge_id, None)
            return None, "expired"
        if c.used_at is not None:
            return None, "used"
        # 先标记使用、再返回：即使后续分析抛异常，这个令牌也不会被二次使用
        c.used_at = time.time()
        return c, None


def stats() -> dict:
    """运行状态，供 /api/capabilities 之外的排查使用。"""
    with _lock:
        _purge_locked()
        return {"pending": len(_store), "ttl_seconds": TTL_SECONDS}


def clear() -> int:
    with _lock:
        n = len(_store)
        _store.clear()
        return n
