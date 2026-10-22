import argparse
from .launcher import run


def main():
    parser = argparse.ArgumentParser(description="启动本地 Transformer 学习网页")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    arguments = parser.parse_args()
    run(port=arguments.port, open_browser=not arguments.no_browser)


if __name__ == "__main__":
    main()
