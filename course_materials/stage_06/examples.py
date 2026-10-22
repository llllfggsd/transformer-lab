"""注意力机制：按 # %% 分段学习。"""

# %% 为什么需要注意力：按内容取信息
# 固定平均会给每个位置相同权重；注意力让查询决定读取哪些内容。Query 表示当前要找的信息，Key 用于匹配，Value 是被加权取出的内容。点积受向量方向和长度共同影响，不等于一般情况下的余弦相似度。
import torch
from torch import nn
import math
torch.set_num_threads(1)
torch.manual_seed(42)
query = torch.tensor([[1., 0.]])
keys = torch.tensor([[1., 0.], [0., 1.]])
values = torch.tensor([[10., 0.], [0., 20.]])
print("点积分数", query @ keys.T)
print("固定平均", values.mean(dim=0))

# %% 手算缩放点积注意力
# 公式 Attention(Q,K,V)=softmax(QKᵀ/√d_k)V。这里未缩放的两个分数是 1 与 0，缩放后是 1/√2 与 0；先算概率，再算两行 Value 的加权和。Softmax 沿 Key 轴进行。
scores = query @ keys.T / math.sqrt(query.shape[-1])
weights = torch.softmax(scores, dim=-1)
output = weights @ values
print("缩放分数", scores)
print("注意力权重", weights)
print("输出", output)
print("权重和", weights.sum(dim=-1))

# %% 自注意力的 Q/K/V 投影
# 自注意力的 Q、K、V 都来自同一序列，但使用不同的可训练投影矩阵。输入 (B,T,D)，分数形状 (B,T,T)，每个位置可以根据内容读取整个序列。
x = torch.randn(2, 4, 8)
q_proj, k_proj, v_proj = nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 8)
q, k, v = q_proj(x), k_proj(x), v_proj(x)
scores = q @ k.transpose(-2, -1) / math.sqrt(8)
weights = scores.softmax(dim=-1)
context = weights @ v
print(q.shape, scores.shape, context.shape)

# %% 因果掩码：不能偷看未来
# 自回归预测中，第 i 个位置只能看到 0 到 i。上三角矩阵中对角线以上的位置设为 True，再填 -inf，Softmax 后这些权重为零。这里约定 True 表示屏蔽。
length = x.shape[1]
causal_mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
causal_weights = scores.masked_fill(causal_mask, float("-inf")).softmax(dim=-1)
print(causal_mask)
print(causal_weights[0])
print("第一个位置只看自己", causal_weights[0, 0])

# %% Padding 掩码与广播
# Padding 屏蔽的是无效 Key，形状 (B,T) 扩展为 (B,1,T) 与分数广播。不能把一个 Query 对应的所有 Key 都设为 -inf，否则普通 Softmax 会得到 NaN。这里每个样本至少有一个有效位置。
valid = torch.tensor([[True, True, True, True], [True, True, False, False]])
blocked = causal_mask.unsqueeze(0) | (~valid).unsqueeze(1)
masked_weights = scores.masked_fill(blocked, float("-inf")).softmax(dim=-1)
print("第二个样本的权重", masked_weights[1])
print("无效 Key 的权重为零", masked_weights[1, :, 2:].abs().sum().item())

# %% 多头：拆分与合并
# D=8 拆为 H=2 个头，每头维度 4。先 reshape 为 (B,T,H,Dh)，再 transpose 成 (B,H,T,Dh)。各头独立计算注意力，最后合并并进行输出投影。
batch_size, length, dimension = x.shape
heads = 2
def split_heads(tensor):
    return tensor.reshape(batch_size, length, heads, dimension // heads).transpose(1, 2)
qh, kh, vh = split_heads(q), split_heads(k), split_heads(v)
head_weights = (qh @ kh.transpose(-2, -1) / math.sqrt(dimension // heads)).softmax(dim=-1)
per_head = head_weights @ vh
merged = per_head.transpose(1, 2).contiguous().reshape(batch_size, length, dimension)
out_proj = nn.Linear(dimension, dimension)
print(qh.shape, head_weights.shape, out_proj(merged).shape)

# %% 使用 PyTorch 多头注意力
# batch_first=True 接收 (B,T,D)。nn.MultiheadAttention 的布尔 attn_mask 和 key_padding_mask 都用 True 表示禁止关注。不同 API 可能相反，例如直接使用 SDPA 时布尔 Mask 的含义不同，不能机械复用。
attention = nn.MultiheadAttention(8, 2, dropout=0., batch_first=True)
result, attention_weights = attention(x, x, x, attn_mask=causal_mask, key_padding_mask=~valid, average_attn_weights=False)
print("输出", result.shape)
print("各头权重", attention_weights.shape)

# %% 验证未来信息没有泄漏
# 把最后一个 Token 改得很大，如果因果掩码正确，前面位置的输出应保持不变。这个检查比只打印 Mask 更能验证实际数据流。Dropout 设为零，避免随机性干扰。
attention.eval()
changed = x.clone()
changed[:, -1] += 100
with torch.no_grad():
    before = attention(x, x, x, attn_mask=causal_mask, need_weights=False)[0]
    after = attention(changed, changed, changed, attn_mask=causal_mask, need_weights=False)[0]
print("前缀一致", torch.allclose(before[:, :-1], after[:, :-1], atol=1e-5))
