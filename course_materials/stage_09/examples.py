"""动手实践：按 # %% 分段学习。"""

# %% 准备可复现的小语料
# 用字符词表与连续数据切片构造任务。语料重复且很小，只用于验证训练链路；验证段也包含重复句式，因此低损失不能证明具备真实语言理解能力。
import torch
from torch import nn
import math
torch.set_num_threads(1)
torch.manual_seed(42)
corpus = "春风又绿江南岸。明月何时照我还。" * 32
characters = sorted(set(corpus))
stoi = {character: index for index, character in enumerate(characters)}
data = torch.tensor([stoi[c] for c in corpus])
split = int(len(data) * 0.8)
train_data, validation_data = data[:split], data[split:]
block_size = 12
print("词表与数据", len(characters), train_data.shape, validation_data.shape)

# %% 亲手实现因果自注意力
# 一次 Linear 得到 Q/K/V，拆头后计算缩放分数，屏蔽未来位置，再合并各头。这个模块直接参与后面的训练，不只用于打印形状。
class CausalSelfAttention(nn.Module):
    def __init__(self, dimension, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dimension, 3 * dimension)
        self.output = nn.Linear(dimension, dimension)
    def forward(self, x):
        batch, length, dimension = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [a.reshape(batch, length, self.heads, dimension // self.heads).transpose(1, 2) for a in (q, k, v)]
        scores = q @ k.transpose(-2, -1) / math.sqrt(dimension // self.heads)
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1)
        weights = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
        context = (weights @ v).transpose(1, 2).contiguous().reshape(batch, length, dimension)
        return self.output(context)
print(CausalSelfAttention(24, 3)(torch.randn(2, 12, 24)).shape)

# %% 搭建可训练的 Transformer 文本模型
# Token Embedding 加可学习的位置 Embedding，经过一个 Pre-Norm 注意力/FFN 模块，再映射到词表。输出 (B,T,V)。同一位置的标签是下一个字符。
class TinyLanguageModel(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocabulary_size, 24)
        self.position_embedding = nn.Embedding(12, 24)
        self.norm1 = nn.LayerNorm(24)
        self.norm2 = nn.LayerNorm(24)
        self.attention = CausalSelfAttention(24, 3)
        self.ffn = nn.Sequential(nn.Linear(24, 96), nn.GELU(), nn.Linear(96, 24))
        self.head = nn.Linear(24, vocabulary_size)
    def forward(self, ids):
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.token_embedding(ids) + self.position_embedding(positions)
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return self.head(x)
model = TinyLanguageModel(len(characters))
print("模型参数", sum(p.numel() for p in model.parameters()))

# %% 构造小批量训练样本
# 每个随机起点取 T+1 个字符，前 T 个作输入，后 T 个作标签。torch.randint 的上界不包含自身，所以有效起点为 [0,len-T)。
def get_batch(sequence, batch_size=8):
    starts = torch.randint(len(sequence) - block_size, (batch_size,))
    x = torch.stack([sequence[i:i + block_size] for i in starts])
    y = torch.stack([sequence[i + 1:i + block_size + 1] for i in starts])
    return x, y
x, y = get_batch(train_data)
print(x.shape, y.shape)
print("目标右移", torch.equal(x[:, 1:], y[:, :-1]))

# %% 训练小型文本生成模型
# 交叉熵把 (B,T,V) 展平成 (B*T,V)。固定 100 次小更新，打印训练前后损失。不追求实际写作质量，重点是观察完整训练流程和参数改变。
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
model.train()
history = []
for step in range(100):
    x, y = get_batch(train_data)
    optimizer.zero_grad()
    logits = model(x)
    loss = nn.functional.cross_entropy(logits.reshape(-1, len(characters)), y.reshape(-1))
    loss.backward()
    optimizer.step()
    history.append(loss.item())
print("训练首尾损失", history[0], history[-1])

# %% 验证损失与困惑度
# 困惑度 exp(平均交叉熵) 可用来观察模型对 Token 的不确定性，但不同词表和数据集的困惑度不能直接比较。本例验证数据含重复模式，不能当作真实泛化评测。
model.eval()
with torch.no_grad():
    vx, vy = get_batch(validation_data)
    validation_logits = model(vx)
    validation_loss = nn.functional.cross_entropy(validation_logits.reshape(-1, len(characters)), vy.reshape(-1))
print("验证损失", validation_loss.item())
print("困惑度", validation_loss.exp().item())

# %% 保存并使用已有的训练权重
# 把配置、词表和训练好的权重一并保存，再实例化并加载。此时 restored 使用已有训练参数，不是随机新模型。真实预训练模型的下载实践另见本阶段 optional_pretrained.py。
checkpoint = {"state_dict": model.state_dict(), "characters": characters}
torch.save(checkpoint, "stage09_tiny_lm.pt")
loaded = torch.load("stage09_tiny_lm.pt", weights_only=True)
restored = TinyLanguageModel(len(loaded["characters"]))
restored.load_state_dict(loaded["state_dict"])
restored.eval()
with torch.no_grad():
    print("重新加载一致", torch.allclose(model(vx), restored(vx)))

# %% 自回归文本生成
# 从一个提示字符出发，每次读取最后位置 logits，选出下一个 ID 再接回输入。上下文超过 12 个字符时裁掉旧前缀。本例使用贪心选择，采样时可替换为 multinomial。
generated = torch.tensor([[stoi["春"]]])
with torch.no_grad():
    for step in range(24):
        logits = restored(generated[:, -block_size:])[:, -1]
        next_id = logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_id], dim=1)
print("生成文本", "".join(characters[index] for index in generated[0].tolist()))
