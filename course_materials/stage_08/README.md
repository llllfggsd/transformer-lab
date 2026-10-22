# 第 8 阶段：常见模型与训练方式

用无需下载权重的微型 BERT、GPT 和 T5 区分结构与训练目标。

共 7 段示例、3 道独立练习和 3 道验收题。

## 使用方式

推荐在网页中逐段学习；也可以打开 `lesson.ipynb` 或运行 `examples.py`。独立练习填写 `exercises.py`，完成后再查看 `answers.py`。

在项目根目录安装本阶段依赖：

```powershell
python -m pip install -r course_materials/stage_08/requirements.txt
python course_materials/stage_08/examples.py
python course_materials/stage_08/exercises.py
```

示例使用 CPU，不依赖 GPU 或下载大模型。第 8 阶段模型从配置随机初始化，只演示架构；第 9 阶段会实际训练一个小模型，文件写在运行时工作目录。网页中的学习记录只由网页的实际验收更新，运行文件版不会自动完成账号进度。

## 本章内容

- Encoder-only：微型 BERT：BERT 使用双向 Encoder，适合表示学习、分类和掩码预测。这里从配置创建真实 BertModel，但权重随机，不能据此判断文本含义。所有配置都很小，不会下载模型。首次导入依赖可能较慢。
- Decoder-only：微型 GPT：GPT 类模型使用因果注意力做下一个 Token 预测。GPT2LMHeadModel 在 labels=input_ids 时会在内部错开预测与标签；不要再手动右移一次。下面的模型同样是随机初始化。
- Encoder–Decoder：微型 T5：T5 的 Encoder 读取输入，Decoder 自回归生成目标并通过交叉注意力读取输入表示。给 labels 时库会创建右移后的 Decoder 输入。原始 Transformer 和 T5 都属于这一家族，但细节不同。
- 预训练任务：掩码预测：预训练先从大量数据学习通用参数。BERT 的掩码语言建模仅在选定位置计算损失；这里只演示把一个位置替换成特殊 ID，不实现完整的随机替换策略。双向模型可以使用左右上下文。
- 微调与冻结参数：微调是在已有模型参数上继续训练。下面只演示冻结 Encoder、训练分类头的机制；由于 Encoder 尚未预训练，这不是可用的情感分类器。真实迁移应加载经过训练的权重。
- 温度、Top-k 与采样：温度 T>0 用 logits/T 改变概率分布；T 越低越集中。Top-k 只保留分数最高的 k 个候选，再归一化并采样。采样控制选择方式，不会让未经训练的模型获得知识。
- 自回归生成与 LLM 的关系：每次用已有 Token 预测一个新 Token，再把它接回上下文。Decoder-only 是常见 LLM 架构，但 Transformer 本身不等于大语言模型：参数规模、数据与训练过程同样关键。下面只观察随机小模型的生成流程。

## 阶段验收

1. 从配置创建 GPT 模型意味着什么？

- 已经有语言能力
- 随机权重，需要训练
- 自动下载完整预训练权重

2. 哪种结构包含读取 Encoder memory 的交叉注意力？

- Encoder–Decoder
- 所有 Decoder-only
- 只有 Embedding

3. 降低正温度通常会怎样改变分布？

- 必然更随机
- 删除所有概率
- 使高分候选更集中

## 参考资料

- [官方参考 1](https://huggingface.co/docs/transformers/model_doc/bert)
- [官方参考 2](https://huggingface.co/docs/transformers/model_doc/gpt2)
- [官方参考 3](https://huggingface.co/docs/transformers/model_doc/t5)
