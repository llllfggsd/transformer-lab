"""第一阶段 A：Python 基础。按 # %% 分段阅读，也可以直接运行整份脚本。"""

# %% 1. 变量：给一个值起名字
learner = "小林"
study_minutes = 30
learning_rate = 0.01
is_beginner = True

print("学习者：", learner)
print("学习分钟数：", study_minutes, "类型：", type(study_minutes).__name__)
print("学习率：", learning_rate, "是否初学：", is_beginner)
print(f"你好，{learner}，今天计划学习 {study_minutes} 分钟。")

# 思考：30 与 "30" 的类型相同吗？把 study_minutes 改为 45 后再运行。

# %% 2. 条件判断：根据条件执行不同分支，缩进表示代码归属
score = 82
if score >= 90:
    level = "非常熟练"
elif score >= 60:
    level = "继续练习"
else:
    level = "先回顾示例"
print("当前建议：", level)

# %% 3. 列表与循环：保存多个值，并逐个处理
scores = [70, 85, 92]
scores.append(88)
print("第一个分数：", scores[0])  # 下标从 0 开始
print("最后一个分数：", scores[-1])
print("前两个分数：", scores[:2])  # 切片右端不包含在内

total = 0
for index, value in enumerate(scores):
    total += value
    print(f"第 {index + 1} 次练习：{value}")
print("平均分：", total / len(scores))

countdown = 3
while countdown > 0:
    print("剩余练习轮数：", countdown)
    countdown -= 1  # 如果遗漏更新，循环就可能无法结束

passed_scores = [value for value in scores if value >= 80]
print("至少 80 分的成绩：", passed_scores)
print("range(3) 生成：", list(range(3)))

# %% 4. 字典：用键查找对应的值
record = {"name": learner, "stage": 1, "scores": scores}
record["finished"] = False
print("学习阶段：", record["stage"])
print("缺少的键可以使用默认值：", record.get("note", "暂无备注"))
for key, value in record.items():
    print(key, "->", value)

# %% 5. 函数：输入参数，返回结果
def average(values):
    """计算一个非空数值列表的平均值。"""
    if len(values) == 0:
        raise ValueError("不能计算空列表的平均值")
    return sum(values) / len(values)


print("用函数算平均分：", average(scores))

# 输入不符合约定时，可以抛出异常；调用方可以捕获并处理。
try:
    average([])
except ValueError as error:
    print("捕获到预期的输入错误：", error)

# %% 6. 类：把数据和操作数据的方法放在一起
class StudyProgress:
    """学习进度；self 指向当前创建的对象。"""

    def __init__(self, name):
        self.name = name
        self.completed_topics = []  # 每个对象拥有自己的列表

    def complete(self, topic):
        self.completed_topics.append(topic)

    def summary(self):
        return f"{self.name} 已完成 {len(self.completed_topics)} 个主题"


progress = StudyProgress(learner)
progress.complete("变量与循环")
progress.complete("函数")
print(progress.summary())
print("已完成的主题：", progress.completed_topics)

# %% 7. 文件读写：把内存中的数据保存到磁盘，再读回来
import json
from pathlib import Path

# 脚本使用自身所在目录；Notebook 放在同一目录，通常以该目录作为工作目录。
lesson_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
# 也支持从项目根目录启动 Notebook 内核。
if (lesson_dir / "stage_01_python").is_dir():
    lesson_dir = lesson_dir / "stage_01_python"
data_dir = lesson_dir / "data"
data_dir.mkdir(exist_ok=True)
record_path = data_dir / "learning_record.json"

saved_record = {"name": progress.name, "completed_topics": progress.completed_topics}
with record_path.open("w", encoding="utf-8") as file:
    json.dump(saved_record, file, ensure_ascii=False, indent=2)

with record_path.open("r", encoding="utf-8") as file:
    loaded_record = json.load(file)
print("从文件读回的数据：", loaded_record)
print("文件位置：", record_path.resolve())

# 试一试：增加一个 completed_topics 项，再写入并读取，观察 JSON 文件变化。
