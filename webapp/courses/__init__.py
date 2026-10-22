"""可运行的分阶段课程，讲解、练习、验收与参考答案共用这一份来源。"""

from textwrap import dedent


def example(title, description, code):
    return {"title": title, "description": description, "code": dedent(code).strip(), "kind": "example", "test": "", "hint": "先预测输出和张量形状，再运行验证。空白及字符串符号不计入核对，变量、数值与代码结构仍需正确。"}


def practice(title, description, signature, solution, tests):
    return {"title": title, "description": description, "code": dedent(signature).strip(), "kind": "practice", "solution": dedent(solution).strip(), "test": dedent(tests).strip(), "hint": "根据题目手敲自己的解法。测试会检查结果、形状和边界输入；不要求与参考答案相同。"}


def quiz(question, options, answer):
    return {"question": question, "options": options, "answer": answer}


def course(number, title, description, tag, examples, practices, questions, requirements=None, sources=None):
    lessons = []
    for kind, items in [("example", examples), ("practice", practices)]:
        for index, item in enumerate(items, start=1):
            lessons.append({**item, "id": f"s{number}-{kind}-{index:02d}", "stage": number, "filename": f"stage_{number:02d}_{kind}.py"})
    return {"number": number, "title": title, "description": description, "tag": tag, "lessons": lessons, "quiz": [{k: v for k, v in item.items() if k != "answer"} for item in questions], "answers": [item["answer"] for item in questions], "requirements": requirements or ["numpy"], "sources": sources or []}
