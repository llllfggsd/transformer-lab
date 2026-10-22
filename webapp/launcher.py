"""供 Python 入口与命令行共用的本地启动逻辑。"""

import asyncio
import sys
import threading
import webbrowser

import uvicorn

from .server import create_app


def run(port=8765, open_browser=True):
    """启动本地服务；服务就绪后再打开浏览器，停止时清理内核。"""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    url = f"http://127.0.0.1:{port}"
    print(f"\nTransformer Lab: {url}\n停止运行或按 Ctrl+C 关闭服务与 Python 内核。\n", flush=True)
    server = uvicorn.Server(uvicorn.Config(
        create_app(), host="127.0.0.1", port=port, loop="asyncio", log_level="warning",
    ))
    stopped = threading.Event()

    def open_when_ready():
        while not stopped.wait(0.1):
            if server.started:
                try:
                    if not webbrowser.open(url):
                        print(f"请手动打开浏览器访问 {url}", flush=True)
                except Exception:
                    print(f"请手动打开浏览器访问 {url}", flush=True)
                return

    if open_browser:
        threading.Thread(target=open_when_ready, daemon=True, name="open-learning-page").start()
    try:
        server.run()
    finally:
        stopped.set()
