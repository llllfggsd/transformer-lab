"""从网页课程的唯一来源生成文件版教材，保留手工编写的 optional_pretrained.py。"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from webapp.curriculum import COURSES


def build():
    target = ROOT / "course_materials"
    target.mkdir(exist_ok=True)
    catalog = ["# Transformer 十阶段课程\n", "网页中仍须按顺序完成示例、独立练习和验收。文件版可用于复习及在普通 Jupyter 中观察输出。\n", "| 阶段 | 内容 | 示例 / 练习 / 验收 |", "| --- | --- | --- |"]
    for number, course in COURSES.items():
        examples = [item for item in course["lessons"] if item["kind"] == "example"]
        practices = [item for item in course["lessons"] if item["kind"] == "practice"]
        path = "../stage_01_python/README.md" if number == 1 else f"stage_{number:02d}/README.md"
        catalog.append(f"| {number:02d} | [{course['title']}]({path}) | {len(examples)} / {len(practices)} / {len(course['quiz'])} |")
        if number == 1:
            continue
        folder = target / f"stage_{number:02d}"
        folder.mkdir(exist_ok=True)
        prelude = "import math\nimport numpy as np\n" + ("import torch\nfrom torch import nn\ntorch.set_num_threads(1)\n" if "torch" in course["requirements"] else "")
        checks = "\n\ndef run_checks(namespace):\n    passed = 0\n    checks = " + repr([item["test"] for item in practices]) + "\n    for index, code in enumerate(checks, 1):\n        try:\n            exec(code, dict(namespace))\n        except Exception as error:\n            print(f'练习 {index}：需要修改（{type(error).__name__}: {error}）')\n        else:\n            passed += 1\n            print(f'练习 {index}：通过')\n    print(f'完成 {passed}/{len(checks)} 题')\n    return passed == len(checks)\n\n\nif __name__ == '__main__':\n    raise SystemExit(0 if run_checks(globals()) else 1)\n"
        (folder / "examples.py").write_text('"""' + course["title"] + '：按 # %% 分段学习。"""\n\n' + "\n\n".join(f"# %% {item['title']}\n# {item['description']}\n{item['code']}" for item in examples) + "\n", encoding="utf-8")
        (folder / "exercises.py").write_text(prelude + "\n\n" + "\n\n".join(f"# {item['title']}\n# {item['description']}\n{item['code']}" for item in practices) + checks, encoding="utf-8")
        (folder / "answers.py").write_text('"""先独立完成 exercises.py，再阅读参考答案。"""\n' + prelude + "\n\n" + "\n\n".join(item["solution"] for item in practices) + checks, encoding="utf-8")
        (folder / "requirements.txt").write_text("\n".join(course["requirements"]) + "\n", encoding="utf-8")
        cells = [{"cell_type": "markdown", "id": f"s{number}-title", "metadata": {}, "source": f"# 第 {number} 阶段：{course['title']}\n\n{course['description']}\n\n先预测，再运行，最后修改输入验证理解。"}]
        for item in course["lessons"]:
            cells.extend([
                {"cell_type": "markdown", "id": item["id"] + "-note", "metadata": {}, "source": f"## {item['title']}\n\n{item['description']}"},
                {"cell_type": "code", "id": item["id"], "metadata": {}, "execution_count": None, "outputs": [], "source": item["code"]},
            ])
        questions = "## 阶段验收\n\n" + "\n\n".join(f"{index}. {question['question']}\n\n" + "\n".join(f"- {option}" for option in question["options"]) for index, question in enumerate(course["quiz"], 1))
        cells.append({"cell_type": "markdown", "id": f"s{number}-quiz", "metadata": {}, "source": questions})
        notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3"}}, "cells": cells}
        (folder / "lesson.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        sources = "\n".join(f"- [官方参考 {index}]({url})" for index, url in enumerate(course["sources"], 1))
        optional = "\n## 真实预训练模型实践\n\n`optional_pretrained.py` 使用真实的英文情感分类模型。首次运行需要联网下载权重，不属于网页必须通过的代码段；请预留下载时间与磁盘空间。\n\n```powershell\npython course_materials/stage_09/optional_pretrained.py\n```\n\n已缓存模型时可以添加 `--local-only`，不连接网络。[模型说明](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)。\n" if number == 9 else ""
        description = f"# 第 {number} 阶段：{course['title']}\n\n{course['description']}\n\n共 {len(examples)} 段示例、{len(practices)} 道独立练习和 {len(course['quiz'])} 道验收题。\n\n## 使用方式\n\n推荐在网页中逐段学习；也可以打开 `lesson.ipynb` 或运行 `examples.py`。独立练习填写 `exercises.py`，完成后再查看 `answers.py`。\n\n在项目根目录安装本阶段依赖：\n\n```powershell\npython -m pip install -r course_materials/stage_{number:02d}/requirements.txt\npython course_materials/stage_{number:02d}/examples.py\npython course_materials/stage_{number:02d}/exercises.py\n```\n\n示例使用 CPU，不依赖 GPU 或下载大模型。第 8 阶段模型从配置随机初始化，只演示架构；第 9 阶段会实际训练一个小模型，文件写在运行时工作目录。网页中的学习记录只由网页的实际验收更新，运行文件版不会自动完成账号进度。\n\n## 本章内容\n\n" + "\n".join(f"- {item['title']}：{item['description']}" for item in examples) + "\n\n" + questions + optional + ("\n\n## 参考资料\n\n" + sources if sources else "") + "\n"
        (folder / "README.md").write_text(description, encoding="utf-8")
    (target / "README.md").write_text("\n".join(catalog) + "\n", encoding="utf-8")
    print("Generated material for stages 02–10.")


if __name__ == "__main__":
    build()
