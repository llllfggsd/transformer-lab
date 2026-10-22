# 第 4 阶段：深度学习与 PyTorch

用自动求导、模型和训练循环完成一个小型分类器。

共 8 段示例、3 道独立练习和 3 道验收题。

## 使用方式

推荐在网页中逐段学习；也可以打开 `lesson.ipynb` 或运行 `examples.py`。独立练习填写 `exercises.py`，完成后再查看 `answers.py`。

在项目根目录安装本阶段依赖：

```powershell
python -m pip install -r course_materials/stage_04/requirements.txt
python course_materials/stage_04/examples.py
python course_materials/stage_04/exercises.py
```

示例使用 CPU，不依赖 GPU 或下载大模型。第 8 阶段模型从配置随机初始化，只演示架构；第 9 阶段会实际训练一个小模型，文件写在运行时工作目录。网页中的学习记录只由网页的实际验收更新，运行文件版不会自动完成账号进度。

## 本章内容

- 张量与自动求导：Tensor 是 PyTorch 的数值容器。requires_grad=True 让系统记录计算图；backward() 把标量损失的梯度累加到叶子张量的 .grad。此课程统一使用 CPU，固定种子便于重现。
- 前向传播、反向传播与清零：前向计算 y=wx+b，反向计算参数对损失的导数。梯度默认累加，所以重复 backward 前要清零。优化器的 zero_grad() 也是解决这个问题。
- 神经元、激活函数与多层网络：线性层执行 xWᵀ+b。多层线性层中间加入非线性激活，才能表示比单个线性映射更复杂的关系。ReLU 会把负数变成零；GELU 更平滑。
- 用类搭建分类模型：继承 nn.Module 并在 __init__ 注册层，在 forward 定义数据流。分类器输出原始 logits，CrossEntropyLoss 内部完成归一化，不需要先做 Softmax。
- Batch、Epoch 与 DataLoader：Batch 是一次更新使用的样本数，Epoch 是遍历完整训练集一次。固定训练/验证划分；每轮打乱训练顺序有助于减少固定顺序的影响。
- 完成训练循环：训练一步依次执行清梯度、前向、损失、反向、优化器更新。model.train() 设置训练模式；学习率决定步长，不能与 Epoch 数混为一谈。
- 验证与推理：model.eval() 切换 Dropout、BatchNorm 等层的行为；torch.no_grad() 关闭求导记录。两者作用不同，推理时通常一起用。这里的准确率只来自小型合成验证集。
- 保存与重新加载参数：state_dict 保存已注册参数。加载时需要建立相同结构，再读入参数；仅加载权重后用 eval 模式推理。此处文件属于当前账号的工作目录。

## 阶段验收

1. 训练一步中哪项在 backward 之后？

- optimizer.step()
- 定义训练集
- optimizer.zero_grad() 必须只在之后

2. CrossEntropyLoss 通常接收什么？

- 先手动 Softmax 的概率
- 原始 logits 和整数类别标签
- 预测类别的 argmax

3. eval() 和 no_grad() 的关系是什么？

- 完全相同
- 都修改模型权重
- 分别控制层行为和梯度记录

## 参考资料

- [官方参考 1](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
