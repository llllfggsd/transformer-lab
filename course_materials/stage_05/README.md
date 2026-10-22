# 第 5 阶段：文本如何变成模型输入

从字符分词到 Embedding、Padding 和下一个 Token 的训练目标。

共 7 段示例、3 道独立练习和 3 道验收题。

## 使用方式

推荐在网页中逐段学习；也可以打开 `lesson.ipynb` 或运行 `examples.py`。独立练习填写 `exercises.py`，完成后再查看 `answers.py`。

在项目根目录安装本阶段依赖：

```powershell
python -m pip install -r course_materials/stage_05/requirements.txt
python course_materials/stage_05/examples.py
python course_materials/stage_05/exercises.py
```

示例使用 CPU，不依赖 GPU 或下载大模型。第 8 阶段模型从配置随机初始化，只演示架构；第 9 阶段会实际训练一个小模型，文件写在运行时工作目录。网页中的学习记录只由网页的实际验收更新，运行文件版不会自动完成账号进度。

## 本章内容

- Token 与分词粒度：Token 是分词器输出的单位，可以是字符、词或子词。这里用字符分词便于手算，它不是 BPE。中文字符通常不是完整的词；真实模型的分词规则由训练好的分词器决定。
- 词表、特殊 Token 与 ID：词表把 Token 映射到整数，ID 的数值大小不表示语义远近。保留 PAD=0、UNK=1；未知字符回退到 UNK，避免索引越界。
- Embedding 查表：Embedding 是形状 (词表大小,向量维度) 的可训练矩阵。输入整数 ID，输出对应的行；输入 (B,T) 会变成 (B,T,D)。padding_idx=0 让 PAD 对应行不通过普通 Embedding 梯度更新。
- 变长序列与 Padding：同一批次中的序列补到相同长度。attention_mask 的 1 表示有效 Token；PyTorch MultiheadAttention 的 key_padding_mask=True 却表示屏蔽，使用前需要明确约定。
- 语言模型：右移一个 Token：自回归语言模型在每个位置预测下一个 Token。训练时把同一序列错开一位得到 x 与 y；输入包含当前 Token，但不能看到未来的目标 Token。
- 逐 Token 的 logits 与损失：模型输出 (B,T,V)，其中 V 是词表大小。CrossEntropyLoss 常将前两个维度展平；PAD 标签可以使用 ignore_index 排除，避免模型把大量补齐位置当作任务。
- 上下文长度与截断：上下文窗口限制一次可使用的历史长度。生成时可以只保留最近一段上下文，但被裁掉的信息无法继续参与本次注意力计算。Padding 不会凭空增加真实信息。

## 阶段验收

1. Embedding 的输入通常是什么？

- 浮点概率
- 整数 Token ID
- 原始 Python 字符串

2. 输入 ID 为 (B,T)，Embedding 维度为 D，输出形状是什么？

- (B,T,D)
- (D,T)
- (B,D)

3. 下一个 Token 训练目标怎样构造？

- 输入和标签完全相同
- 把所有标签设为 PAD
- 序列右移对应一位
