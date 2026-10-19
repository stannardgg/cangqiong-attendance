"""前端交互的真实浏览器端到端测试（Chrome 假摄像头 + CDP 读 DOM）。

为什么需要它：后端接口已用 HTTP 测过，但前端改写后「探测→挑战→采集→校验→提示」
这条链路从未真实跑过。JS 里一个拼写错误就会让学生完全签不了到。

做法：
  1. 把一张真实人脸照片压成 Y4M 视频（Chrome 的假摄像头可以直接播文件）；
  2. Chrome 以 headless 启动，加载页面并自动点「开始签到」；
  3. 用 CDP 每 400ms 读一次 #attStatus，直到出现预期文案；
  4. 同时收集 Runtime.exceptionThrown 与 console error。

预期结果：视频里没有人做动作 → 云端校验必然返回 action_missing →
状态栏出现「没有检测到动作」，这证明整条链路真的跑通了。

用法：先启动后端，再运行
    ./.venv/Scripts/python.exe tools/browser_e2e.py http://127.0.0.1:8011
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
PHOTO = os.path.join(BASE, "data", "test", "real", "lena.jpg")
ROOT = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8011"
DEBUG_PORT = 9333
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

W, H, FPS, FRAMES = 480, 360, 30, 240


def make_y4m(path):
    img = cv2.imdecode(np.fromfile(PHOTO, dtype=np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (W, H))
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)   # 直接得到 I420 内存布局
    with open(path, "wb") as f:
        f.write(f"YUV4MPEG2 W{W} H{H} F{FPS}:1 Ip A1:1 C420mpeg2\n".encode())
        for _ in range(FRAMES):
            f.write(b"FRAME\n")
            f.write(yuv.tobytes())
    return path


def api(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(ROOT + path, data=data,
                                 headers={"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read() or b"null")


class CDP:
    """极简 CDP 客户端：发命令、收结果，事件收进 self.events。"""

    def __init__(self, ws):
        self.ws = ws
        self._id = 0
        self.events = []

    async def send(self, method, params=None):
        self._id += 1
        mid = self._id
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method} 失败: {msg['error']}")
                return msg.get("result", {})
            if "method" in msg:                 # 事件（console / exception）
                self.events.append(msg)

    async def eval(self, expr):
        r = await self.send("Runtime.evaluate",
                            {"expression": expr, "returnByValue": True, "awaitPromise": True})
        return r.get("result", {}).get("value")


async def drive(page_ws, course_id, timeout=45):
    from websockets.asyncio.client import connect as ws_connect

    errors = []
    async with ws_connect(page_ws, max_size=None) as ws:
        cdp = CDP(ws)
        await cdp.send("Runtime.enable")
        await cdp.send("Log.enable")

        # 等页面初始化（课程列表 / 能力集）
        for _ in range(40):
            ready = await cdp.eval(
                "!!document.querySelector('#attCourse') && "
                "document.querySelector('#attCourse').options.length > 1")
            if ready:
                break
            await asyncio.sleep(0.25)

        await cdp.eval(f"""
            (function(){{
              const sel = document.querySelector('#attCourse');
              sel.value = '{course_id}';
              document.querySelector('#btnStartAtt').click();
              return sel.value;
            }})()
        """)

        status, seen, last = "", [], ""
        t0 = time.time()
        while time.time() - t0 < timeout:
            snap = await cdp.eval("""JSON.stringify({
                s: document.querySelector('#attStatus')?.textContent || '',
                banner: !document.querySelector('#challengeBox')?.hidden,
                streak: typeof probeStreak === 'number' ? probeStreak : null,
                attempting: typeof attAttempting === 'boolean' ? attAttempting : null,
                running: typeof attRunning === 'boolean' ? attRunning : null,
                cooldown: typeof attCooldownUntil === 'number' ? attCooldownUntil - Date.now() : null,
            })""")
            d = json.loads(snap or "{}")
            status = d.get("s", "")
            line = (f"    [{time.time()-t0:5.1f}s] s={status!r} streak={d.get('streak')} "
                    f"att={d.get('attempting')} run={d.get('running')} cd={d.get('cooldown')} "
                    f"banner={d.get('banner')}")
            if line != last:
                last = line
                seen.append(status)
                print(line)
            # 动作类失败现在只显示服务端给的那句「该怎么做」，措辞会随指令变化，
            # 所以按关键词集合判断（张一次 / 张两次 / 通用兜底）。
            if any(k in status for k in ("没有检测到动作", "请把嘴张大", "请先闭上嘴",
                                         "验证请求失败", "无法打开摄像头", "通信失败", "采集失败")):
                break
            await asyncio.sleep(0.4)

        conn = {"events": cdp.events[-40:]}
        return status, seen, conn


def main():
    if not os.path.exists(CHROME):
        print("找不到 Chrome，跳过浏览器测试")
        return 0

    tmp = tempfile.mkdtemp(prefix="cq_e2e_")
    y4m = os.path.join(tmp, "face.y4m")
    profile = os.path.join(tmp, "profile")
    print(f"[0] 生成假摄像头视频 {W}x{H} × {FRAMES} 帧 → {os.path.getsize(make_y4m(y4m))//1024//1024}MB")

    course = api("POST", "/api/courses", {"name": "__E2E课程__", "class_name": ""})
    cid = course["id"]
    print(f"[0] 临时课程 id={cid}")

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
            f"--remote-debugging-port={DEBUG_PORT}",
            f"--user-data-dir={profile}",
            "--use-fake-ui-for-media-stream",
            "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-video-capture={y4m}",
            "--autoplay-policy=no-user-gesture-required",
            ROOT + "/",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 等待调试端口可用
        page_ws = None
        for _ in range(40):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{DEBUG_PORT}/json", timeout=1) as r:
                    targets = json.loads(r.read())
                pages = [t for t in targets if t.get("type") == "page"]
                if pages:
                    page_ws = pages[0]["webSocketDebuggerUrl"]
                    break
            except Exception:
                pass
            time.sleep(0.3)
        if not page_ws:
            print("  [FAIL] 无法连接 Chrome 调试端口")
            return 1

        print("[1] 启动签到，观察状态栏")
        status, seen, conn = asyncio.run(drive(page_ws, cid))
        print(f"    最终状态: {status}")

        exc = [e for e in conn["events"]
               if e.get("method") == "Runtime.exceptionThrown"]
        cerr = [e for e in conn["events"]
                if e.get("method") == "Runtime.consoleAPICalled"
                and e.get("params", {}).get("type") == "error"]
        for e in (exc + cerr)[:5]:
            print(f"    浏览器错误: {json.dumps(e, ensure_ascii=False)[:300]}")

        check("摄像头成功打开（脱离了「启动摄像头」阶段）",
              "无法打开摄像头" not in status and bool(seen), f"{len(seen)} 条状态变化")
        check("走到了动作校验（整轮出现过动作指令浮层与校验结论）",
              any(k in s for s in seen for k in ("已检测到人脸", "请把嘴张大", "请先闭上嘴",
                                                 "没有检测到动作")),
              f"{len(seen)} 条状态")
        check("最终给出明确失败原因",
              any(k in status for k in ("请把嘴张大", "请先闭上嘴", "没有检测到动作")), status)
        check("无 JS 运行时异常", not exc and not cerr, f"异常 {len(exc)} / 控制台错误 {len(cerr)}")
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
        shutil.rmtree(tmp, ignore_errors=True)
        try:                      # 无论成败都清掉临时课程
            api("DELETE", f"/api/courses/{cid}")
        except Exception:
            pass

    print(f"\n结果: {passed}/{total} 通过")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
