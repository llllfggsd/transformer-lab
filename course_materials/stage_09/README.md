# 第 9 阶段：动手实践

从自注意力模块出发，训练、保存并重新加载一个小型文本生成模型。

共 8 段示例、3 道独立练习和 3 道验收题。

## 使用方式

推荐在网页中逐段学习；也可以打开 `lesson.ipynb` 或运行 `examples.py`。独立练习填写 `exercises.py`，完成后再查看 `answers.py`。

在项目根目录安装本阶段依赖：

```powershell
python -m pip install -r course_materials/stage_09/requirements.txt
python course_materials/stage_09/examples.py
python course_materials/stage_09/exercises.py
```

示例使用 CPU，不依赖 GPU 或下载大模型。第 8 阶段模型从配置随机初始化，只演示架构；第 9 阶段会实际训练一个小模型，文件写在运行时工作目录。网页中的学习记录只由网页的实际验收更新，运行文件版不会自动完成账号进度。

## 本章内容

- 准备可复现的小语料：用字符词表与连续数据切片构造任务。语料重复且很小，只用于验证训练链路；验证段也包含重复句式，因此低损失不能证明具备真实语言理解能力。
- 亲手实现因果自注意力：一次 Linear 得到 Q/K/V，拆头后计算缩放分数，屏蔽未来位置，再合并各头。这个模块直接参与后面的训练，不只用于打印形状。
- 搭建可训练的 Transformer 文本模型：Token Embedding 加可学习的位置 Embedding，经过一个 Pre-Norm 注意力/FFN 模块，再映射到词表。输出 (B,T,V)。同一位置的标签是下一个字符。
- 构造小批量训练样本：每个随机起点取 T+1 个字符，前 T 个作输入，后 T 个作标签。torch.randint 的上界不包含自身，所以有效起点为 [0,len-T)。
- 训练小型文本生成模型：交叉熵把 (B,T,V) 展平成 (B*T,V)。固定 100 次小更新，打印训练前后损失。不追求实际写作质量，重点是观察完整训练流程和参数改变。
- 验证损失与困惑度：困惑度 exp(平均交叉熵) 可用来观察模型对 Token 的不确定性，但不同词表和数据集的困惑度不能直接比较。本例验证数据含重复模式，不能当作真实泛化评测。
- 保存并使用已有的训练权重：把配置、词表和训练好的权重一并保存，再实例化并加载。此时 restored 使用已有训练参数，不是随机新模型。真实预训练模型的下载实践另见本阶段 optional_pretrained.py。
- 自回归文本生成：从一个提示字符出发，每次读取最后位置 logits，选出下一个 ID 再接回输入。上下文超过 12 个字符时裁掉旧前缀。本例使用贪心选择，采样时可替换为 multinomial。

## 阶段验收

1. 训练输入 x 和目标 y 应该是什么关系？

- x 的后续一个 Token 对应 y
- 完全无关
- y 必须全为零

2. 加载 state_dict 前需要什么？

- 删除词表
- 建立匹配的模型结构
- 重新训练一遍

3. 重复小语料上的低验证损失能证明通用语言能力吗？

- 可以
- 只要损失为零就可以
- 不能，需要独立真实数据评估
## 真实预训练模型实践

`optional_pretrained.py` 使用真实的英文情感分类模型。首次运行需要联网下载权重，不属于网页必须通过的代码段；请预留下载时间与磁盘空间。

```powershell
python course_materials/stage_09/optional_pretrained.py
```

已缓存模型时可以添加 `--local-only`，不连接网络。[模型说明](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)。


## 参考资料

- [官方参考 1](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
