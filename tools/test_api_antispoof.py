"""端到端接口测试（对运行中的后端发真实 HTTP 请求）。

用法：先启动后端，再运行
    ./.venv/Scripts/python.exe tools/test_api_antispoof.py http://127.0.0.1:8011

覆盖：
  · /probe 不写签到
  · /challenge 签发随机动作
  · /verify 用「同一张照片重复 16 帧」提交 → 必须被拒（action_missing）
  · 同一令牌二次提交 → used
  · 伪造令牌 → not_found
  · 数据库签到记录数量在失败路径上不得变化
"""
import base64
import hashlib
import json
import os
import sys
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(BASE, "data", "attendance.db")
PHOTO = os.path.join(BASE, "data", "test", "real", "lena.jpg")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8011"


def post(path, body):
    req = urllib.request.Request(
        ROOT + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def get(path):
    with urllib.request.urlopen(ROOT + path, timeout=30) as r:
        return json.loads(r.read())


def b64_small():
    """复刻前端 captureSmall 的参数（448 宽 / q0.62），保证测的就是真实链路。"""
    img = cv2.imdecode(np.fromfile(PHOTO, dtype=np.uint8), cv2.IMREAD_COLOR)
    h = int(img.shape[0] * 448 / img.shape[1])
    img = cv2.resize(img, (448, h))
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 62])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def db_fingerprint():
    with open(DB, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    total = passed = 0

    def check(name, cond, extra=""):
        nonlocal total, passed
        total += 1
        passed += 1 if cond else 0
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")

    course_id = get("/api/courses")[0]["id"]
    before = db_fingerprint()
    print(f"课程 id={course_id}  DB md5={before[:8]}")

    # ---- 1. probe ----
    frame = b64_small()
    print("\n[1] 单帧探测 /probe")
    st, p = post("/api/attendance/probe", {"image": frame, "course_id": course_id})
    print(f"    status={st} {p}")
    check("接口可用", st == 200)
    check("识别到一张人脸", p.get("faces") == 1, f"faces={p.get('faces')}")
    check("判定可以开始验证", p.get("ready") is True)

    # ---- 2. challenge ----
    print("\n[2] 申请挑战 /challenge")
    st, c = post("/api/attendance/challenge", {"course_id": course_id})
    print(f"    status={st} action={c.get('action')} {c.get('instruction')}")
    check("签发成功", st == 200 and "challenge_id" in c)
    check("动作已随机指定", c.get("action") in ("mouth", "mouth2"))

    # ---- 3. verify：同一张照片重复 16 帧 ----
    print("\n[3] 提交「同一张照片 ×16 帧」（模拟举照片代签）")
    frames = [frame] * 16
    size_kb = len(json.dumps(frames)) / 1024
    st, v = post("/api/attendance/verify", {
        "course_id": course_id, "challenge_id": c["challenge_id"], "frames": frames,
    })
    m = v.get("metrics", {})
    print(f"    status={st} 请求体 {size_kb:.0f}KB")
    print(f"    ok={v.get('ok')} reason={v.get('reason')} hint={v.get('hint')}")
    print(f"    活体={m.get('liveness_median')} 相似度={v.get('similarity')} "
          f"帧数={m.get('frames')} 人脸宽={m.get('face_width_px')}")
    check("照片被拒", v.get("ok") is False)
    check("拒绝原因是动作未完成", v.get("reason") == "action_missing", f"→ {v.get('reason')}")
    check("单帧活体确实被骗过（0.5 以上）", (m.get("liveness_median") or 0) >= 0.5,
          f"liveness={m.get('liveness_median')}")

    # ---- 4. 令牌一次性 ----
    print("\n[4] 同一令牌再提交一次")
    st, v2 = post("/api/attendance/verify", {
        "course_id": course_id, "challenge_id": c["challenge_id"], "frames": frames,
    })
    print(f"    ok={v2.get('ok')} reason={v2.get('reason')}")
    check("重放被拒", v2.get("ok") is False and v2.get("reason") == "used")

    # ---- 5. 伪造令牌 ----
    print("\n[5] 伪造令牌")
    st, v3 = post("/api/attendance/verify", {
        "course_id": course_id, "challenge_id": "forged-token-123456", "frames": frames,
    })
    print(f"    ok={v3.get('ok')} reason={v3.get('reason')}")
    check("伪造令牌被拒", v3.get("reason") == "not_found")

    # ---- 6. 诊断接口 ----
    print("\n[6] 诊断接口 /api/attendance/diagnostics")
    d = get("/api/attendance/diagnostics")
    print(f"    models={d.get('models')} challenges={d.get('challenges')}")
    check("模型状态齐全", all(d.get("models", {}).values()))

    # ---- 7. 数据未被污染 ----
    print("\n[7] 失败路径不得写库")
    check("DB 逐字节未变", db_fingerprint() == before)

    print(f"\n结果: {passed}/{total} 通过")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
