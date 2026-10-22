"""补齐八处 TODO，然后运行本文件自查。参考答案放在另一份文件中。"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def score_label(score):
    """题 1：分数 >= 60 返回字符串 '通过'，否则返回 '继续练习'。"""
    # TODO：使用条件判断。
    raise NotImplementedError("请完成题 1")


def passing_scores(scores):
    """题 2：返回所有 >= 60 的分数，保留原顺序；空列表返回空列表。"""
    # TODO：使用循环或列表推导式。
    raise NotImplementedError("请完成题 2")


def mean_score(scores):
    """题 3：返回平均分；如果 scores 为空列表，抛出 ValueError。"""
    # TODO：结合函数、条件判断、sum 和 len。
    raise NotImplementedError("请完成题 3")


def count_words(text):
    """题 4：按空白分隔文本，返回 {单词: 出现次数}。区分大小写。

    例如 'I love Python I' -> {'I': 2, 'love': 1, 'Python': 1}。
    提示：text.split() 会把连续空白当作分隔符；本题不处理标点。
    """
    # TODO：使用字典累计计数。
    raise NotImplementedError("请完成题 4")


class TopicTracker:
    """题 5：用类记录已完成的主题，允许重复添加同一个主题。"""

    def __init__(self, name):
        """保存 self.name，并创建属于当前对象的空列表 self.topics。"""
        # TODO：补齐构造方法和下面的 complete 方法，合计一题。
        raise NotImplementedError("请完成题 5 的构造方法")

    def complete(self, topic):
        """把 topic 加入 self.topics，返回累计完成的主题数量。"""
        raise NotImplementedError("请完成题 5 的 complete 方法")


def save_and_load_json(record, path):
    """题 6：用 UTF-8 将字典 record 写入 path，再读回并返回字典。

    path 是 pathlib.Path，父目录已存在；record 只包含 JSON 支持的值。
    提示：使用 json.dump/load；ensure_ascii=False 便于直接阅读中文。
    """
    # TODO：分别用 with 打开文件进行写入和读取。
    raise NotImplementedError("请完成题 6")


def row_means(x):
    """题 7：x 是形状 (样本数, 特征数) 的二维数组，返回每行平均值。

    输出必须是一维 NumPy 数组，形状为 (样本数,)。输入至少有一行一列。
    """
    # TODO：选择正确的 axis。
    raise NotImplementedError("请完成题 7")


def linear_forward(x, weights, bias):
    """题 8：返回 x @ weights + bias，使用 NumPy 数组运算。

    x: (batch_size, input_dim)，weights: (input_dim, output_dim)，
    bias: (output_dim,)，输出: (batch_size, output_dim)。输入形状合法。
    """
    # TODO：区分矩阵乘法和逐元素乘法。
    raise NotImplementedError("请完成题 8")


# 以下是学习用自查，不需要修改。参考答案也会使用同一组检查。
def run_checks(answers):
    """answers 是包含上面函数和类的字典；返回是否全部通过。"""

    def check_1():
        function = answers["score_label"]
        assert function(59) == "继续练习", "59 分应该继续练习"
        assert function(60) == "通过", "60 分是通过的边界"
        assert function(100) == "通过", "100 分应该通过"

    def check_2():
        function = answers["passing_scores"]
        source = [59, 90, 60, 40, 80]
        assert function(source) == [90, 60, 80], "检查筛选条件与顺序"
        assert source == [59, 90, 60, 40, 80], "请不要修改输入列表"
        assert function([]) == [], "空列表应该返回空列表"

    def check_3():
        function = answers["mean_score"]
        assert function([1, 2]) == 1.5, "平均值应保留小数"
        assert function([80]) == 80, "单个分数的平均值就是它自己"
        try:
            function([])
        except ValueError:
            return
        raise AssertionError("空列表应该抛出 ValueError")

    def check_4():
        function = answers["count_words"]
        assert function("I love Python I") == {"I": 2, "love": 1, "Python": 1}
        assert function("  你好\tPython\n你好 ") == {"你好": 2, "Python": 1}
        assert function("") == {}, "空文本应返回空字典"

    def check_5():
        tracker = answers["TopicTracker"]("小林")
        other = answers["TopicTracker"]("小周")
        assert tracker.name == "小林"
        assert tracker.topics == []
        assert tracker.complete("函数") == 1
        assert tracker.complete("函数") == 2
        assert tracker.topics == ["函数", "函数"]
        assert other.topics == [], "两个对象不应该共享 topics 列表"

    def check_6():
        record = {"name": "小林", "scores": [60, 90], "finished": False}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            assert answers["save_and_load_json"](record, path) == record
            assert path.exists(), "请实际保存文件"
            with path.open(encoding="utf-8") as file:
                assert json.load(file) == record, "磁盘文件内容不正确"

    def check_7():
        x = np.array([[1, 3, 5], [2, 4, 6]], dtype=np.float32)
        result = answers["row_means"](x)
        assert isinstance(result, np.ndarray), "请返回 NumPy 数组"
        assert result.shape == (2,), "每行一个值，输出形状应该是 (2,)"
        np.testing.assert_allclose(result, [3, 4])
        np.testing.assert_allclose(answers["row_means"](np.array([[2, 4]])), [3])

    def check_8():
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        weights = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
        result = answers["linear_forward"](x, weights, np.array([0.5, -0.5]))
        assert isinstance(result, np.ndarray), "请返回 NumPy 数组"
        assert result.shape == (2, 2), "检查输出的样本数和特征数"
        np.testing.assert_allclose(result, [[4.5, 4.5], [10.5, 10.5]])
        # 再用不同形状检查一次，避免只适用于上面的示例。
        result = answers["linear_forward"](
            np.array([[2, 3]]), np.array([[1, 2, 3], [4, 5, 6]]), np.array([1, 0, -1])
        )
        np.testing.assert_allclose(result, [[15, 19, 23]])

    checks = [check_1, check_2, check_3, check_4, check_5, check_6, check_7, check_8]
    passed = 0
    for number, check in enumerate(checks, start=1):
        try:
            check()
        except NotImplementedError as error:
            print(f"题 {number}：尚未完成（{error}）")
        except Exception as error:
            print(f"题 {number}：需要修改（{type(error).__name__}: {error}）")
        else:
            passed += 1
            print(f"题 {number}：通过")
    print(f"\n完成 {passed}/{len(checks)} 题。")
    if passed == len(checks):
        print("请继续完成 README 中的口头解释，再确认进入下一阶段。")
    return passed == len(checks)


if __name__ == "__main__":
    raise SystemExit(0 if run_checks(globals()) else 1)
