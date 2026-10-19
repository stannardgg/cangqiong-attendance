"""手工补签（兜底通道）接口测试。

会临时创建「课程 + 学生」，测完删除，不污染真实数据。
覆盖：
  · 补签写入成功，记录 status='manual'（前端显示为「补签」）
  · 重复补签不产生第二条（幂等）
  · 手机/公网来源（带代理头）被 403 拦住
  · 非法 ID 返回 404
  · 清理后无残留
"""
import json
import sys
import urllib.error
import urllib.request

ROOT = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8011"

_total = 0
_passed = 0


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
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def main():
    _, course = call("POST", "/api/courses", {"name": "__临时测试课程__", "class_name": ""})
    _, stu = call("POST", "/api/students",
                  {"name": "__临时学生__", "student_no": "__TMP001__", "class_name": ""})
    cid, sid = course["id"], stu["id"]
    print(f"临时课程 id={cid}  临时学生 id={sid}")

    try:
        print("\n[1] 本机补签")
        st, r = call("POST", "/api/attendance/manual", {"course_id": cid, "student_id": sid})
        print(f"    status={st} result={r.get('result')} {r.get('name')} {r.get('check_in_at')}")
        check("补签成功", st == 200 and r.get("result") == "attended")

        print("\n[2] 签到记录里的状态")
        _, recs = call("GET", f"/api/attendance?course_id={cid}")
        status = recs[0]["status"] if recs else None
        print(f"    status={status} 条数={len(recs)}")
        check("标记为 manual", status == "manual" and len(recs) == 1)

        print("\n[3] 重复补签（幂等）")
        _, r2 = call("POST", "/api/attendance/manual", {"course_id": cid, "student_id": sid})
        _, recs2 = call("GET", f"/api/attendance?course_id={cid}")
        print(f"    result={r2.get('result')} 条数={len(recs2)}")
        check("不产生重复记录", r2.get("result") == "already" and len(recs2) == 1)

        print("\n[4] 手机/公网来源（带代理头）必须被拒")
        st, r3 = call("POST", "/api/attendance/manual",
                      {"course_id": cid, "student_id": sid},
                      headers={"X-Forwarded-For": "1.2.3.4"})
        print(f"    status={st} detail={str(r3)[:70]}")
        check("远程补签被 403 拦截", st == 403)

        print("\n[5] 不存在的课程 / 学生")
        st1, e1 = call("POST", "/api/attendance/manual", {"course_id": 999999, "student_id": sid})
        st2, e2 = call("POST", "/api/attendance/manual", {"course_id": cid, "student_id": 999999})
        check("非法 ID 返回 404", st1 == 404 and st2 == 404)
    finally:
        call("DELETE", f"/api/courses/{cid}")
        call("DELETE", f"/api/students/{sid}")
        print("\n已清理临时数据")

    _, courses = call("GET", "/api/courses")
    left = [c for c in courses if c["name"].startswith("__临时")]
    check("无临时数据残留", not left)

    print(f"\n结果: {_passed}/{_total} 通过")
    return 0 if _passed == _total else 1


if __name__ == "__main__":
    sys.exit(main())
