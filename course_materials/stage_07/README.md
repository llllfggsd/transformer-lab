# 第 7 阶段：Transformer 的完整结构

把位置、注意力、前馈网络、残差和归一化串成完整的数据流。

共 7 段示例、3 道独立练习和 3 道验收题。

## 使用方式

推荐在网页中逐段学习；也可以打开 `lesson.ipynb` 或运行 `examples.py`。独立练习填写 `exercises.py`，完成后再查看 `answers.py`。

在项目根目录安装本阶段依赖：

```powershell
python -m pip install -r course_materials/stage_07/requirements.txt
python course_materials/stage_07/examples.py
python course_materials/stage_07/exercises.py
```

示例使用 CPU，不依赖 GPU 或下载大模型。第 8 阶段模型从配置随机初始化，只演示架构；第 9 阶段会实际训练一个小模型，文件写在运行时工作目录。网页中的学习记录只由网页的实际验收更新，运行文件版不会自动完成账号进度。

## 本章内容

- 位置编码：给顺序一个坐标：不加入位置信息时，普通自注意力本身缺乏位置标签。正弦位置编码使用不同频率的 sin/cos，输出 (T,D)，再广播到 (B,T,D)。这里 D 为偶数。
- 前馈网络 FFN：FFN 在每个 Token 上独立应用相同的两层映射，先扩展维度、激活，再压回 D。它混合特征维度，注意力则让不同 Token 交换信息。
- 残差与 LayerNorm：残差连接把子层输出加回输入，要求两者形状相同。LayerNorm(8) 沿最后 8 个特征归一化。原论文采用 Post-Norm，很多现代模型采用 Pre-Norm，两者计算顺序不同。
- 亲手串联一个 Pre-Norm 模块：顺序为 x + Attention(LN(x))，然后 x + FFN(LN(x))。注意力可以接收因果掩码，同一个模块可用不同 Mask 控制可见范围。
- Encoder 编码器：编码器使用双向自注意力，把源序列转成上下文表示。Padding Key 需要被屏蔽。下面使用 PyTorch 的一层 Encoder；D=8，H=2，FFN 隐层=32。
- Decoder 与交叉注意力：原始 Decoder 先做因果自注意力，再用目标序列作 Query、Encoder memory 作 Key/Value 做交叉注意力。GPT 类 Decoder-only 通常不包含这一步交叉注意力。
- 完整 Encoder–Decoder 数据流：源 ID→Embedding→位置→Encoder memory；右移后的目标 ID→Embedding→位置→Decoder→词表 logits。nn.Transformer 不包含词表 Embedding 与最终分类头，需要自行添加。

## 阶段验收

1. 交叉注意力的 Key/Value 通常来自哪里？

- 只有未来目标
- Encoder memory
- 损失函数

2. FFN 主要在哪个维度做变换？

- 每个 Token 的特征维度
- 只交换 Batch
- 只改变序列长度

3. nn.Transformer 是否自动包含 Token Embedding 和词表输出头？

- 包含全部
- 只包含词表输出头
- 两者都需要自行添加

## 参考资料

- [官方参考 1](https://arxiv.org/abs/1706.03762)
- [官方参考 2](https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
