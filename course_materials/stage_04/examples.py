"""深度学习与 PyTorch：按 # %% 分段学习。"""

# %% 张量与自动求导
# Tensor 是 PyTorch 的数值容器。requires_grad=True 让系统记录计算图；backward() 把标量损失的梯度累加到叶子张量的 .grad。此课程统一使用 CPU，固定种子便于重现。
import torch
from torch import nn
torch.set_num_threads(1)
torch.manual_seed(42)
x = torch.tensor([2., 3.], requires_grad=True)
loss = (x ** 2).sum()
loss.backward()
print("损失", loss.item(), "梯度", x.grad)

# %% 前向传播、反向传播与清零
# 前向计算 y=wx+b，反向计算参数对损失的导数。梯度默认累加，所以重复 backward 前要清零。优化器的 zero_grad() 也是解决这个问题。
weight = torch.tensor(3., requires_grad=True)
bias = torch.tensor(1., requires_grad=True)
prediction = weight * 2 + bias
loss = (prediction - 5) ** 2
loss.backward()
print("预测与损失", prediction.item(), loss.item())
print("参数梯度", weight.grad.item(), bias.grad.item())
weight.grad.zero_()
bias.grad.zero_()

# %% 神经元、激活函数与多层网络
# 线性层执行 xWᵀ+b。多层线性层中间加入非线性激活，才能表示比单个线性映射更复杂的关系。ReLU 会把负数变成零；GELU 更平滑。
values = torch.tensor([-2., -1., 0., 1., 2.])
print("ReLU", torch.relu(values))
print("GELU", nn.functional.gelu(values))
network = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
print("批量输出", network(torch.ones(4, 2)).shape)
print("参数数量", sum(p.numel() for p in network.parameters()))

# %% 用类搭建分类模型
# 继承 nn.Module 并在 __init__ 注册层，在 forward 定义数据流。分类器输出原始 logits，CrossEntropyLoss 内部完成归一化，不需要先做 Softmax。
class SmallClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, 12), nn.Tanh(), nn.Linear(12, 2))
    def forward(self, features):
        return self.layers(features)
model = SmallClassifier()
print(model)
print(model(torch.zeros(3, 2)).shape)

# %% Batch、Epoch 与 DataLoader
# Batch 是一次更新使用的样本数，Epoch 是遍历完整训练集一次。固定训练/验证划分；每轮打乱训练顺序有助于减少固定顺序的影响。
from torch.utils.data import TensorDataset, DataLoader
features = torch.randn(160, 2)
labels = (features[:, 0] + features[:, 1] > 0).long()
train_data = TensorDataset(features[:128], labels[:128])
loader = DataLoader(train_data, batch_size=32, shuffle=True)
validation_x, validation_y = features[128:], labels[128:]
print("每轮批次数", len(loader))
batch_x, batch_y = next(iter(loader))
print(batch_x.shape, batch_y.shape)

# %% 完成训练循环
# 训练一步依次执行清梯度、前向、损失、反向、优化器更新。model.train() 设置训练模式；学习率决定步长，不能与 Epoch 数混为一谈。
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
loss_fn = nn.CrossEntropyLoss()
model.train()
for epoch in range(20):
    total_loss = 0.
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
print("最后一轮平均损失", total_loss / len(loader))

# %% 验证与推理
# model.eval() 切换 Dropout、BatchNorm 等层的行为；torch.no_grad() 关闭求导记录。两者作用不同，推理时通常一起用。这里的准确率只来自小型合成验证集。
model.eval()
with torch.no_grad():
    logits = model(validation_x)
    predicted = logits.argmax(dim=-1)
    accuracy = (predicted == validation_y).float().mean()
print("验证准确率", accuracy.item())
print("单个样本的概率", torch.softmax(logits[0], dim=-1))

# %% 保存与重新加载参数
# state_dict 保存已注册参数。加载时需要建立相同结构，再读入参数；仅加载权重后用 eval 模式推理。此处文件属于当前账号的工作目录。
torch.save(model.state_dict(), "stage04_classifier.pt")
restored = SmallClassifier()
restored.load_state_dict(torch.load("stage04_classifier.pt", weights_only=True))
restored.eval()
with torch.no_grad():
    print("加载结果一致", torch.allclose(model(validation_x), restored(validation_x)))
