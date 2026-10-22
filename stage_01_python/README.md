# 第一阶段：Python 与编程基础

目标：能读懂并修改一段 Python 程序，能操作二维 NumPy 数组，能在 Notebook 中逐格观察运行结果。这一阶段先熟悉代码，不要求理解神经网络。

建议分三次学习：先 Python，再 NumPy，最后独立完成练习；按自己的速度推进。

## 准备环境

在下载后的项目根目录打开 PowerShell。已有环境可以直接使用；如果需要独立环境，执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r stage_01_python\requirements.txt
```

下面命令中的 `python` 指你安装了上述依赖的 Python。使用独立环境时，把 `python` 替换成 `.\.venv\Scripts\python.exe` 即可，不必激活环境。

## 跟着示例学习

推荐打开 Notebook：

```powershell
python -m jupyterlab stage_01_python\01_python_numpy.ipynb
```

选择 Python 内核，依次选中代码格，按 `Shift+Enter` 运行。先预测输出，再运行验证，再修改一个值重新运行。变量会保留在内核中；遇到状态混乱时，重启内核并从上往下运行全部单元格。

不使用 Notebook 也可以运行等价的讲解脚本：

```powershell
python stage_01_python\01_python_basics.py
python stage_01_python\02_numpy_basics.py
```

- `01_python_basics.py`：变量、条件、循环、函数、类、列表、字典、文件读写。
- `02_numpy_basics.py`：数组创建、形状、索引、广播、逐元素乘法、矩阵乘法、批处理。
- `01_python_numpy.ipynb`：以上两份脚本的逐格学习版，包含思考问题。

文件读写示例会在本阶段的 `data/` 下生成 `learning_record.json`，重复运行会更新这一个演示文件。

## 独立练习

编辑 `03_practice.py`，只需要补齐八处 `TODO` 对应的函数体，替换 `raise NotImplementedError(...)`。函数下面的自查代码保留即可。

```powershell
python stage_01_python\03_practice.py
```

第一次运行显示“尚未完成”是正常的。每完成一题就运行一次，自查会继续检查其他题目。这些检查覆盖本阶段的基本要求，通过后还应读懂自己的代码。

确实卡住时再看 `04_reference_answers.py`，也可以运行它查看参考解法的自查结果：

```powershell
python stage_01_python\04_reference_answers.py
```

## 确认掌握

准备进入第二阶段前，请完成：

1. 八道练习全部通过，并能用自己的话解释每个函数。
2. 解释列表 `[1, 2] * 2` 与 NumPy 数组 `np.array([1, 2]) * 2` 的区别。
3. 解释 `(2, 3)` 的数组与 `(3, 4)` 的数组做矩阵乘法，结果为什么是 `(2, 4)`。
4. 把示例中的二维输入增加一行，预测哪些输出的形状会改变，再运行检查。

网页中完成示例、独立练习与三道验收题后，即可解锁第二阶段。文件版的第二阶段已提供在 [数学基础](../course_materials/stage_02/README.md)。也可以在聊天里回复“第一阶段完成”并附上第 2、3 题的解释，或贴出卡住的代码与报错。
