"""一个本机 Jupyter 内核；只通过标准消息协议执行代码和收集结果。"""

import os
from pathlib import Path
from queue import Empty
import sys
import time

from jupyter_client import KernelManager


class PythonKernel:
    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.manager = None
        self.client = None
        self.position = None

    def start(self):
        self.close()
        runtime = self.workdir / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["IPYTHONDIR"] = str(runtime / "ipython")
        environment["JUPYTER_RUNTIME_DIR"] = str(runtime)
        environment["PYTHONIOENCODING"] = "utf-8"
        self.manager = KernelManager(connection_file=str(runtime / "kernel.json"))
        # 使用启动网页的 Python，避免系统 kernelspec 指向另一个缺少 NumPy 的环境。
        self.manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        try:
            self.manager.start_kernel(cwd=str(self.workdir), env=environment)
            self.client = self.manager.blocking_client()
            self.client.start_channels()
            self.client.wait_for_ready(timeout=30)
            self.position = 0
        except Exception:
            self.close()
            raise

    def execute(self, code, timeout=20):
        message_id = self.client.execute(code, allow_stdin=False, stop_on_error=True)
        outputs = []
        output_size = 0
        deadline = time.monotonic() + timeout
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                try:
                    message = self.client.get_iopub_msg(timeout=min(remaining, 1))
                except Empty:
                    if not self.manager.is_alive():
                        raise RuntimeError("Python 内核已退出，请重新运行当前段。")
                    continue
                if message.get("parent_header", {}).get("msg_id") != message_id:
                    continue
                kind = message["msg_type"]
                content = message["content"]
                text = None
                if kind == "stream":
                    text = content["text"]
                elif kind in ("execute_result", "display_data"):
                    text = content.get("data", {}).get("text/plain", "")
                elif kind == "error":
                    text = "\n".join(content.get("traceback", [])) or content.get("evalue", "运行出错")
                elif kind == "clear_output":
                    outputs.clear()
                if text is not None and output_size < 60000:
                    text = text[:60000 - output_size]
                    outputs.append({"type": "error" if kind == "error" else "text", "text": text})
                    output_size += len(text)
                if kind == "status" and content["execution_state"] == "idle":
                    break
            # IOPub 的 idle 和 shell 的 execute_reply 是独立通道，两个都必须确认。
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                try:
                    reply = self.client.get_shell_msg(timeout=min(remaining, 1))
                except Empty:
                    continue
                if reply.get("parent_header", {}).get("msg_id") == message_id:
                    content = reply["content"]
                    return {"ok": content.get("status") == "ok", "outputs": outputs, "execution_count": content.get("execution_count")}
        except TimeoutError:
            # 重建内核，避免超时代码的迟到消息污染下一次执行。
            self.close()
            return {"ok": False, "outputs": [{"type": "error", "text": f"运行超过 {timeout} 秒，已停止内核。请检查循环，然后重试；先前通过的代码会自动恢复。"}], "execution_count": None}
        except Exception:
            self.close()
            raise

    def close(self):
        if self.client:
            self.client.stop_channels()
            self.client = None
        if self.manager:
            try:
                self.manager.shutdown_kernel(now=True)
            finally:
                self.manager = None
        self.position = None
