"""前端「签到成功」路径的真实浏览器测试。

上一个脚本（browser_e2e.py）证明的是「照片类画面会被拒」。
本脚本反过来验证**成功之后界面是否正确**：
注册临时人脸 → 喂一段预烘焙了「周期性张嘴」的视频 →
无论服务端要求张一次还是张两次，只要等待几轮就会命中一次 →
检查状态栏出现「签到成功」且数据库真的多了一条记录。

这是判据链路与 UI 的集成测试，不是安全评估：
真实攻击者不会给你一段「有节奏反复张嘴」的素材，这里只是为了让成功分支跑起来。
"""
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import cv2
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from tools.browser_e2e import CHROME, CDP, DEBUG_PORT, FPS, api, PHOTO  # noqa: E402

W, H, FRAMES = 480, 480, 240          # 8 秒；与 lena 原图同比例，避免把脸压扁
ROOT = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8011"


def make_action_y4m(path):
    """生成带周期性张嘴的视频：每 1.2 秒一个循环（张开 0.5s + 闭合 0.7s）。

    节奏由判据反推：
      · 前端约 100ms 采一帧、一次采集 22 帧（约 2.2 秒）；
      · 一次张嘴至少连续 2 帧才算「张开了一次」，两次之间必须有闭合间隙（否则会并成一次）；
      · 指令可能要求「张一次」也可能要求「张两次」，所以视频里 2.2 秒内必须能数出 ≥2 次。
    周期 1.2 秒、张开 0.5 秒 → 任何一个 2.2 秒窗口都能覆盖两次完整的张开。
    """
    from tools.synth import Face

    face = Face(width=W)
    opened = face.mouth_open()
    if face.img.shape[:2] != (H, W):
        opened = cv2.resize(opened, (W, H))
        base = cv2.resize(face.img, (W, H))
    else:
        base = face.img

    cycle = int(1.2 * FPS)
    open_span = (int(0.20 * FPS), int(0.70 * FPS))   # 张开整 0.5 秒

    with open(path, "wb") as f:
        f.write(f"YUV4MPEG2 W{W} H{H} F{FPS}:1 Ip A1:1 C420mpeg2\n".encode())
        for i in range(FRAMES):
            t = i % cycle
            frame = opened if open_span[0] <= t < open_span[1] else base
            f.write(b"FRAME\n")
            f.write(cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420).tobytes())
    return path


async def drive(page_ws, course_id, timeout=75):
    from websockets.asyncio.client import connect as ws_connect

    async with ws_connect(page_ws, max_size=None) as ws:
        cdp = CDP(ws)
        await cdp.send("Runtime.enable")
        for _ in range(40):
            ok = await cdp.eval("!!document.querySelector('#attCourse') && "
                                "document.querySelector('#attCourse').options.length > 1")
            if ok:
                break
            await asyncio.sleep(0.25)
        await cdp.eval(f"""
            (function(){{
              document.querySelector('#attCourse').value = '{course_id}';
              document.querySelector('#btnStartAtt').click();
            }})()
        """)

        status, seen = "", []
        t0 = time.time()
        while time.time() - t0 < timeout:
            status = await cdp.eval("document.querySelector('#attStatus')?.textContent || ''") or ""
            if status != (seen[-1] if seen else None):
                seen.append(status)
                print(f"    [{time.time()-t0:5.1f}s] {status}")
            if "签到成功" in status:
                break
            await asyncio.sleep(0.4)
        exc = [e for e in cdp.events if e.get("method") == "Runtime.exceptionThrown"]
        return status, seen, exc


def main():
    if not os.path.exists(CHROME):
        print("找不到 Chrome，跳过")
        return 0

    # 先用「不开动作」的方式注册人脸：注册接口只做检测 + 特征提取
    raw = cv2.imdecode(np.fromfile(PHOTO, dtype=np.uint8), cv2.IMREAD_COLOR)
    hh = int(raw.shape[0] * 448 / raw.shape[1])
    reg = cv2.resize(raw, (448, hh), interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", reg, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    import base64
    reg_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

    course = api("POST", "/api/courses", {"name": "__E2E成功课程__", "class_name": ""})
    stu = api("POST", "/api/students",
                 {"name": "__E2E成功学生__", "student_no": "__E2E_OK__", "class_name": ""})
    cid, sid = course["id"], stu["id"]
    api("POST", f"/api/students/{sid}/faces", {"image": reg_b64})
    print(f"[0] 临时课程 id={cid} 学生 id={sid}（已注册人脸）")

    tmp = tempfile.mkdtemp(prefix="cq_ok_")
    y4m = os.path.join(tmp, "act.y4m")
    profile = os.path.join(tmp, "profile")
    print(f"[0] 生成动作视频 → {os.path.getsize(make_action_y4m(y4m))//1024//1024}MB")

    proc = None
    total = passed = 0

    def check(name, cond, extra=""):
        nonlocal total, passed
        total += 1
        passed += 1 if cond else 0
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")

    try:
        proc = subprocess.Popen([
            CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
            "--no-first-run", "--no-default-browser-check",
            f"--remote-debugging-port={DEBUG_PORT}", f"--user-data-dir={profile}",
            "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-video-capture={y4m}",
            "--autoplay-policy=no-user-gesture-required",
            ROOT + "/",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        page_ws = None
        for _ in range(40):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{DEBUG_PORT}/json", timeout=1) as r:
                    pages = [t for t in json.loads(r.read()) if t.get("type") == "page"]
                if pages:
                    page_ws = pages[0]["webSocketDebuggerUrl"]
                    break
            except Exception:
                pass
            time.sleep(0.3)
        if not page_ws:
            print("  [FAIL] 无法连接 Chrome 调试端口")
            return 1

        print("[1] 启动签到，等待「签到成功」")
        status, seen, exc = asyncio.run(drive(page_ws, cid))
        print(f"    最终状态: {status}")

        check("界面显示签到成功", "签到成功" in status, status)
        check("显示的是正确的学生姓名", "__E2E成功学生__" in status, status)
        check("无 JS 运行时异常", not exc, f"异常 {len(exc)}")

        recs = api("GET", f"/api/attendance?course_id={cid}")
        print(f"    数据库记录: {len(recs)} 条 {recs[0]['name'] if recs else ''}")
        check("数据库已落账", len(recs) == 1 and recs[0]["name"] == "__E2E成功学生__")
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            api("DELETE", f"/api/courses/{cid}")
            api("DELETE", f"/api/students/{sid}")
        except Exception:
            pass

    print(f"\n结果: {passed}/{total} 通过")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
