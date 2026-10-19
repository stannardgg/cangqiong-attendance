"""苍穹签到系统 — 启动入口。

运行后端服务，并自动打开浏览器访问界面。

监听地址说明：
  · 本机双击 start.bat 运行时，监听 0.0.0.0（所有网卡），
    这样**手机连同一个 WiFi 就能访问** http://<本机局域网IP>:8000 ；
  · 若容器/隧道环境设置了 PORT 环境变量，则沿用该端口，且不自动弹浏览器
    （云端没有图形界面，弹浏览器会报错）。
"""
import os
import socket
import threading
import webbrowser

import uvicorn

PORT_ENV = os.environ.get("PORT")
PORT = int(PORT_ENV) if PORT_ENV else 8000
HOST = os.environ.get("HOST", "0.0.0.0")          # 0.0.0.0 = 局域网内其它设备也能连
LOCAL_URL = f"http://127.0.0.1:{PORT}"


def lan_ip() -> str:
    """取本机在局域网里的 IP。

    不真的发包：UDP connect 只是让系统按路由表挑一张出口网卡，然后读它的地址。
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return ""
    finally:
        s.close()


def open_browser() -> None:
    webbrowser.open(LOCAL_URL)


def print_banner() -> None:
    ip = lan_ip()
    line = "=" * 58
    print(line)
    print("  苍穹签到系统 已启动")
    print(f"  本机访问：{LOCAL_URL}")
    if ip:
        print(f"  手机访问：http://{ip}:{PORT}   （手机需连同一个 WiFi）")
    else:
        print("  手机访问：未检测到局域网 IP（可能没连 WiFi）")
    print(line)
    print("  提示：手机第一次打不开，多半是 Windows 防火墙拦了，")
    print("        右键以管理员身份运行  allow-phone-access.bat  放行一次即可。")
    print(line)


if __name__ == "__main__":
    print_banner()
    if not PORT_ENV:                     # 隧道/云端环境不弹浏览器
        threading.Timer(1.5, open_browser).start()
    uvicorn.run("backend.main:app", host=HOST, port=PORT, log_level="info")
