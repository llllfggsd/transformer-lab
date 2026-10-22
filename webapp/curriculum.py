"""十阶段课程目录；第一阶段沿用原始 ID，兼容已有账号进度。"""

import ast
import json
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE_DIR = ROOT / "stage_01_python"
STAGE_TITLES = [
    "Python 与编程基础", "必要的数学基础", "机器学习基础", "深度学习与 PyTorch",
    "文本与模型输入", "注意力机制", "Transformer 完整结构", "常见模型与训练方式",
    "动手实践", "进阶内容",
]

PRACTICE_TESTS = [
    "assert score_label(59) == '继续练习'\nassert score_label(60) == '通过'\nassert score_label(100) == '通过'",
    "a = [59, 90, 60, 40, 80]\nassert passing_scores(a) == [90, 60, 80]\nassert a == [59, 90, 60, 40, 80]\nassert passing_scores([]) == []",
    "assert mean_score([1, 2]) == 1.5\nassert mean_score([80]) == 80\ntry:\n    mean_score([])\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('空列表应抛出 ValueError')",
    "assert count_words('I love Python I') == {'I': 2, 'love': 1, 'Python': 1}\nassert count_words('  你好\\tPython\\n你好 ') == {'你好': 2, 'Python': 1}\nassert count_words('') == {}",
    "a = TopicTracker('小林')\nb = TopicTracker('小周')\nassert a.name == '小林' and a.topics == []\nassert a.complete('函数') == 1\nassert a.complete('函数') == 2\nassert a.topics == ['函数', '函数']\nassert b.topics == [], '每个对象需要独立的列表'",
    "import json\nfrom pathlib import Path\nfrom tempfile import TemporaryDirectory\nwith TemporaryDirectory() as d:\n    p = Path(d) / 'test.json'\n    data = {'name': '小林', 'scores': [60, 90]}\n    assert save_and_load_json(data, p) == data\n    assert json.loads(p.read_text(encoding='utf-8')) == data",
    "import numpy as np\nr = row_means(np.array([[1, 3, 5], [2, 4, 6]]))\nassert isinstance(r, np.ndarray) and r.shape == (2,)\nnp.testing.assert_allclose(r, [3, 4])\nnp.testing.assert_allclose(row_means(np.array([[2, 4]])), [3])",
    "import numpy as np\nr = linear_forward(np.array([[1, 2, 3], [4, 5, 6]]), np.array([[1, 0], [0, 1], [1, 1]]), np.array([0.5, -0.5]))\nassert isinstance(r, np.ndarray) and r.shape == (2, 2)\nnp.testing.assert_allclose(r, [[4.5, 4.5], [10.5, 10.5]])\nnp.testing.assert_allclose(linear_forward(np.array([[2, 3]]), np.array([[1, 2, 3], [4, 5, 6]]), np.array([1, 0, -1])), [[15, 19, 23]])",
]


def load_lessons():
    notebook = json.loads((STAGE_DIR / "01_python_numpy.ipynb").read_text(encoding="utf-8"))
    lessons = []
    heading = ""
    description = ""
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown" and source.startswith("### "):
            heading, _, description = source[4:].partition("\n\n")
        elif cell["cell_type"] == "code":
            # 注释仍保留在原 Notebook；抄写区域去掉独立注释行，减少无意义输入。
            code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#")).strip()
            lessons.append({
                "id": f"s1-example-{len(lessons) + 1:02d}", "stage": 1,
                "title": heading, "description": description, "code": code,
                "kind": "example", "test": "", "hint": "对照左侧完成代码。空白、注释、引号风格及字符串中的中英文符号不影响核对；变量、数值、文字内容和代码结构仍需正确。",
            })

    source = (STAGE_DIR / "03_practice.py").read_text(encoding="utf-8")
    definitions = [node for node in ast.parse(source).body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name != "run_checks"]
    answer_source = (STAGE_DIR / "04_reference_answers.py").read_text(encoding="utf-8")
    answers = {node.name: ast.get_source_segment(answer_source, node) for node in ast.parse(answer_source).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    practice_titles = ["条件判断", "列表筛选", "平均分与异常", "字典计数", "类与独立对象", "JSON 文件读写", "沿轴求平均", "批量线性运算"]
    for index, node in enumerate(definitions):
        starter = ast.get_source_segment(source, node)
        # 给出接口与题目，不暴露参考答案。
        lessons.append({
            "id": f"s1-practice-{index + 1:02d}", "stage": 1,
            "title": f"练习 {index + 1} · {practice_titles[index]}",
            "description": (ast.get_docstring(node) or "") + "\n\n请手敲定义并完成函数体；用你自己的解法替换 NotImplementedError。需要的模块可以自行 import。",
            "code": starter, "kind": "practice", "test": PRACTICE_TESTS[index], "solution": answers[node.name],
            "hint": "本题按运行结果验收，支持不同解法。测试包括边界输入；失败时查看右侧错误信息。",
        })
    return lessons


LESSONS = load_lessons()
QUIZ = [
    {"question": "[1, 2] * 2 和 np.array([1, 2]) * 2 的结果分别是什么？", "options": ["[2, 4] 和 [2, 4]", "[1, 2, 1, 2] 和 [2, 4]", "[1, 2, 1, 2] 和 [1, 2, 1, 2]"]},
    {"question": "形状 (2, 3) 与 (3, 4) 的二维数组做矩阵乘法，输出形状是什么？", "options": ["(3, 3)", "(2, 3, 4)", "(2, 4)"]},
    {"question": "x 的形状是 (5, 3)，计算每个样本的特征平均值应该使用什么？", "options": ["x.mean(axis=1)，结果形状 (5,)", "x.mean(axis=0)，结果形状 (3,)", "x.mean()，结果是一个数"]},
]
QUIZ_ANSWERS = [1, 2, 0]

# LESSONS/QUIZ 的旧名字继续指向第一阶段，供旧脚本使用。
COURSES = {1: {"number": 1, "title": STAGE_TITLES[0], "description": "熟悉 Python、NumPy 与 Notebook，为后续张量运算打基础。", "tag": "PYTHON FOUNDATIONS", "lessons": LESSONS, "quiz": QUIZ, "answers": QUIZ_ANSWERS, "requirements": ["numpy"], "sources": []}}
for number in range(2, 11):
    COURSE = import_module(f"webapp.courses.stage{number:02d}").COURSE
    COURSES[number] = COURSE

ALL_LESSONS = [lesson for course in COURSES.values() for lesson in course["lessons"]]
LESSON_LOOKUP = {lesson["id"]: (number, index) for number, course in COURSES.items() for index, lesson in enumerate(course["lessons"])}
