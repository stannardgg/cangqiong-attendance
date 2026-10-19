"""签到成功路径的端到端测试（真实 HTTP）。

已有测试只覆盖「照片被拒」的失败路径。本测试补上**成功路径**：
注册一张临时人脸 → 申请挑战 → 提交一段「含正确动作」的合成序列 → 必须真的写进签到记录。

序列用 `tools/synth.py` 生成（在嘴内画暗腔模拟张嘴），
并且按服务端下发的**次数**（张一次 / 张两次）准备素材 ——
次数是随机性所在，做少了必须判不过。

注意：这是**判据链路测试**，不是安全评估。它证明「校验通过后能正确写入签到」，
不能证明真人一定能过。
"""
import base64
import json
import os
import sys
import urllib.error
import urllib.request

import cv2
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from tools.synth import Face, sequence  # noqa: E402

PHOTO = os.path.join(BASE, "data", "test", "real", "lena.jpg")
ROOT = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8011"
N = 22      # 与前端 BURST_FRAMES 保持一致

_total = _passed = 0


def check(name, cond, extra=""):
    global _total, _passed
    _total += 1
    _passed += 1 if cond else 0
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")


def call(method, path, body=None, headers=None):
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(ROOT + path, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def b64(img):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 62])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def main():
    face = Face()
    img = face.img

    def frames_for(action):
        """按服务端下发的动作与次数生成对应序列。"""
        return sequence(face, "mouth2" if action == "mouth2" else "mouth", n=N)

    # ---- 准备：临时学生 + 注册人脸 ----
    _, stu = call("POST", "/api/students",
                  {"name": "__成功路径测试__", "student_no": "__TMP_OK__", "class_name": ""})
    _, course = call("POST", "/api/courses", {"name": "__成功路径课程__", "class_name": ""})
    sid, cid = stu["id"], course["id"]
    print(f"临时学生 id={sid}  临时课程 id={cid}")

    try:
        print("\n[1] 注册人脸（本机允许）")
        st, r = call("POST", f"/api/students/{sid}/faces", {"image": b64(img)})
        print(f"    status={st} faces_total={r.get('faces_total')}")
        check("人脸注册成功", st == 200 and r.get("faces_total") == 1)

        print("\n[2] 申请挑战（服务端随机动作 + 随机次数）")
        st, c = call("POST", "/api/attendance/challenge", {"course_id": cid})
        print(f"    action={c.get('action')} ({c.get('instruction')})")
        check("挑战签发成功", st == 200 and c.get("action") in ("mouth", "mouth2"))

        print("\n[3] 提交含正确动作的序列（附带指令前的基线帧）")
        st, v = call("POST", "/api/attendance/verify", {
            "course_id": cid, "challenge_id": c["challenge_id"],
            "frames": [b64(f) for f in frames_for(c["action"])],
            "ref": b64(img),
        })
        m = v.get("metrics", {})
        print(f"    ok={v.get('ok')} reason={v.get('reason')} action={v.get('action')}")
        print(f"    识别={v.get('name')} 相似度={v.get('similarity')} 活体={m.get('liveness_median')}")

        print("\n[4] 校验通过 → 必须真正写入签到")
        print(f"    ok={v.get('ok')} result={v.get('result')} name={v.get('name')} "
              f"check_in_at={v.get('check_in_at')}")
        check("验证通过", v.get("ok") is True, f"reason={v.get('reason')}")
        check("返回签到成功", v.get("result") == "attended")
        check("识别为正确学生", v.get("name") == "__成功路径测试__")
        check("返回签到时间", bool(v.get("check_in_at")))

        _, recs = call("GET", f"/api/attendance?course_id={cid}")
        print(f"    数据库记录 {len(recs)} 条: status={recs[0]['status'] if recs else None}")
        check("数据库已落一条出勤", len(recs) == 1 and recs[0]["status"] == "present")

        print("\n[5] 指定「张两次嘴」再验一次 → 必须通过动作校验，再返回 already")
        # 挑战内容是服务端随机的，这一步专门抢一个 mouth2，
        # 保证「两次」这条更严的分支也被真实的 HTTP 链路（解码→判据→落库判断）覆盖。
        c3, tries = None, 0
        while tries < 12:
            tries += 1
            _, cand = call("POST", "/api/attendance/challenge", {"course_id": cid})
            if cand.get("action") == "mouth2":
                c3 = cand
                break
        print(f"    抢到 mouth2 用了 {tries} 次 challenge")
        check("能拿到张两次的指令", c3 is not None)
        st, v3 = call("POST", "/api/attendance/verify", {
            "course_id": cid, "challenge_id": c3["challenge_id"],
            "frames": [b64(f) for f in frames_for("mouth2")],
            "ref": b64(img),
        })
        d3 = v3.get("metrics", {}).get("action_detail", {})
        print(f"    ok={v3.get('ok')} reason={v3.get('reason')} result={v3.get('result')} "
              f"count={d3.get('count')}/{d3.get('need_count')}")
        check("两次动作被识别通过", v3.get("reason") != "action_missing",
              f"reason={v3.get('reason')}")
        check("重复签到返回 already", v3.get("ok") is True and v3.get("result") == "already")

        print("\n[6] 指令要求两次、只做一次 → 必须判不过（随机次数的意义）")
        c4, tries = None, 0
        while tries < 12:
            tries += 1
            _, cand = call("POST", "/api/attendance/challenge", {"course_id": cid})
            if cand.get("action") == "mouth2":
                c4 = cand
                break
        if c4 is None:
            print("    没抢到 mouth2，跳过")
        else:
            st, v4 = call("POST", "/api/attendance/verify", {
                "course_id": cid, "challenge_id": c4["challenge_id"],
                "frames": [b64(f) for f in sequence(face, "mouth", n=N)],
                "ref": b64(img),
            })
            d4 = v4.get("metrics", {}).get("action_detail", {})
            print(f"    ok={v4.get('ok')} reason={v4.get('reason')} "
                  f"count={d4.get('count')}/{d4.get('need_count')}")
            check("只做一次被判不过", v4.get("ok") is False and v4.get("reason") == "action_missing")

        _, recs2 = call("GET", f"/api/attendance?course_id={cid}")
        check("签到记录仍为 1 条", len(recs2) == 1)
    finally:
        call("DELETE", f"/api/courses/{cid}")
        call("DELETE", f"/api/students/{sid}")
        print("\n已清理临时学生与课程")

    print(f"\n结果: {_passed}/{_total} 通过")
    return 0 if _passed == _total else 1


if __name__ == "__main__":
    sys.exit(main())
