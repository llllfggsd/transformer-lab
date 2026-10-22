"""直接在 IDE 中运行本文件，即可启动学习网页。"""

from webapp.launcher import run

# 可以在这里修改启动配置。
PORT = 8765
OPEN_BROWSER = True


if __name__ == "__main__":
    run(port=PORT, open_browser=OPEN_BROWSER)
