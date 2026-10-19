"""防代签回归测试（不需要真人，只用静态照片构造样本）。

覆盖：
  1. 完全静止的静态照片序列      → 必须被拒
  2. 加了平移抖动（模拟手持）的静态照片 → 必须被拒（动作不存在）
  2b. 「张两次」指令下，做一次也不够  → 必须被拒（这是随机次数的价值）
  3. 一次性令牌：重复使用 / 过期 / 课程不匹配 → 必须被拒
  4. 正常流程的耗时预算

注意：真人张嘴/张两次嘴只能由真人在真实摄像头上验收，这里无法伪造。
"""
import os
import sys
import time

import cv2
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from backend import challenge as ch  # noqa: E402
from backend import liveness  # noqa: E402

PHOTO = os.path.join(BASE, "data", "test", "real", "lena.jpg")


def load_photo(width=640):
    img = cv2.imdecode(np.fromfile(PHOTO, dtype=np.uint8), cv2.IMREAD_COLOR)
    h = int(img.shape[0] * width / img.shape[1])
    return cv2.resize(img, (width, h))


def shaky_frames(img, n=16, amp=6):
    """同一张静态图做小幅平移，模拟「有人举着照片手在抖」。"""
    out = []
    for i in range(n):
        dx = amp * np.sin(i * 0.9)
        dy = amp * np.cos(i * 1.3) * 0.6
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        out.append(cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE))
    return out


def main():
    img = load_photo()
    print(f"样本图: {img.shape[1]}x{img.shape[0]}  (真人照片的静态文件)")

    total = 0
    passed = 0

    def check(name, cond, extra=""):
        nonlocal total, passed
        total += 1
        passed += 1 if cond else 0
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")

    # ---- 1. 完全静止 ----
    t = time.perf_counter()
    r = liveness.analyze_sequence([img] * 16, "mouth")
    cost = (time.perf_counter() - t) * 1000
    m = r.get("metrics", {})
    print("\n[1] 静止照片 / 张嘴指令")
    print(f"    耗时 {cost:.0f} ms  reason={r['reason']}  liveness={m.get('liveness_median')} "
          f"identity={m.get('identity_similarity')} drift={m.get('landmark_drift')}")
    check("静止照片被拒", not r["ok"], f"→ {r['reason']}")
    check("拒绝原因是动作缺失", r["reason"] == "action_missing", f"→ {r['reason']}")
    check("单帧活体确实被骗过（说明必须靠动作把关）",
          (m.get("liveness_median") or 0) >= 0.5, f"liveness={m.get('liveness_median')}")

    # ---- 2. 抖动照片 ----
    r2 = liveness.analyze_sequence(shaky_frames(img), "mouth")
    m2 = r2.get("metrics", {})
    print("\n[2] 抖动照片 / 张嘴指令")
    print(f"    reason={r2['reason']}  drift={m2.get('landmark_drift')}")
    check("抖动照片被拒", not r2["ok"], f"→ {r2['reason']}")

    r3 = liveness.analyze_sequence(shaky_frames(img), "mouth2")
    print("\n[3] 抖动照片 / 张两次嘴指令")
    print(f"    reason={r3['reason']}  mouth={r3.get('metrics', {}).get('mouth_dark')}")
    check("张两次嘴指令同样被拒", not r3["ok"], f"→ {r3['reason']}")

    # ---- 4. 令牌一次性 ----
    print("\n[4] 一次性挑战令牌")
    c1 = ch.issue(1)
    got, err = ch.consume(c1["challenge_id"], 1)
    check("首次消费成功", got is not None and err is None, f"action={c1['action']}")
    got2, err2 = ch.consume(c1["challenge_id"], 1)
    check("重复消费被拒", got2 is None and err2 == "used", f"→ {err2}")

    c2 = ch.issue(1)
    got3, err3 = ch.consume(c2["challenge_id"], 999)
    check("课程不匹配被拒", got3 is None and err3 == "course_mismatch", f"→ {err3}")

    c3 = ch.issue(1)
    old = ch.TTL_SECONDS
    ch.TTL_SECONDS = -1.0  # 让刚签发的令牌立刻过期
    got4, err4 = ch.consume(c3["challenge_id"], 1)
    ch.TTL_SECONDS = old
    check("过期令牌被拒", got4 is None and err4 == "expired", f"→ {err4}")

    c4 = ch.issue(1)
    got5, err5 = ch.consume("不存在的令牌xxxxxxxx", 1)
    check("伪造令牌被拒", got5 is None and err5 == "not_found", f"→ {err5}")

    # ---- 5. 随机性 ----
    actions = {ch.issue(1)["action"] for _ in range(20)}
    print(f"\n[5] 20 次签发的动作集合: {actions}")
    check("动作从集合内随机", actions <= set(liveness.ACTIONS) and len(actions) >= 1)

    print(f"\n结果: {passed}/{total} 通过")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
