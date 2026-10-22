# 第 6 阶段：注意力机制

从极小的 Q/K/V 手算例子，走到多头注意力和掩码。

共 8 段示例、3 道独立练习和 3 道验收题。

## 使用方式

推荐在网页中逐段学习；也可以打开 `lesson.ipynb` 或运行 `examples.py`。独立练习填写 `exercises.py`，完成后再查看 `answers.py`。

在项目根目录安装本阶段依赖：

```powershell
python -m pip install -r course_materials/stage_06/requirements.txt
python course_materials/stage_06/examples.py
python course_materials/stage_06/exercises.py
```

示例使用 CPU，不依赖 GPU 或下载大模型。第 8 阶段模型从配置随机初始化，只演示架构；第 9 阶段会实际训练一个小模型，文件写在运行时工作目录。网页中的学习记录只由网页的实际验收更新，运行文件版不会自动完成账号进度。

## 本章内容

- 为什么需要注意力：按内容取信息：固定平均会给每个位置相同权重；注意力让查询决定读取哪些内容。Query 表示当前要找的信息，Key 用于匹配，Value 是被加权取出的内容。点积受向量方向和长度共同影响，不等于一般情况下的余弦相似度。
- 手算缩放点积注意力：公式 Attention(Q,K,V)=softmax(QKᵀ/√d_k)V。这里未缩放的两个分数是 1 与 0，缩放后是 1/√2 与 0；先算概率，再算两行 Value 的加权和。Softmax 沿 Key 轴进行。
- 自注意力的 Q/K/V 投影：自注意力的 Q、K、V 都来自同一序列，但使用不同的可训练投影矩阵。输入 (B,T,D)，分数形状 (B,T,T)，每个位置可以根据内容读取整个序列。
- 因果掩码：不能偷看未来：自回归预测中，第 i 个位置只能看到 0 到 i。上三角矩阵中对角线以上的位置设为 True，再填 -inf，Softmax 后这些权重为零。这里约定 True 表示屏蔽。
- Padding 掩码与广播：Padding 屏蔽的是无效 Key，形状 (B,T) 扩展为 (B,1,T) 与分数广播。不能把一个 Query 对应的所有 Key 都设为 -inf，否则普通 Softmax 会得到 NaN。这里每个样本至少有一个有效位置。
- 多头：拆分与合并：D=8 拆为 H=2 个头，每头维度 4。先 reshape 为 (B,T,H,Dh)，再 transpose 成 (B,H,T,Dh)。各头独立计算注意力，最后合并并进行输出投影。
- 使用 PyTorch 多头注意力：batch_first=True 接收 (B,T,D)。nn.MultiheadAttention 的布尔 attn_mask 和 key_padding_mask 都用 True 表示禁止关注。不同 API 可能相反，例如直接使用 SDPA 时布尔 Mask 的含义不同，不能机械复用。
- 验证未来信息没有泄漏：把最后一个 Token 改得很大，如果因果掩码正确，前面位置的输出应保持不变。这个检查比只打印 Mask 更能验证实际数据流。Dropout 设为零，避免随机性干扰。

## 阶段验收

1. 注意力权重沿哪个维度归一化？

- Key 所在维度
- Batch 维度
- 任意维度都相同

2. 因果掩码需要屏蔽什么？

- 所有历史位置
- 未来位置
- 当前位置及之前所有位置

3. 多头分数矩阵通常是什么形状？

- (B,D)
- (T,H)
- (B,H,T,T)

## 参考资料

- [官方参考 1](https://arxiv.org/abs/1706.03762)
- [官方参考 2](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
