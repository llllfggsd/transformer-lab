"""文本如何变成模型输入：按 # %% 分段学习。"""

# %% Token 与分词粒度
# Token 是分词器输出的单位，可以是字符、词或子词。这里用字符分词便于手算，它不是 BPE。中文字符通常不是完整的词；真实模型的分词规则由训练好的分词器决定。
import torch
from torch import nn
torch.set_num_threads(1)
torch.manual_seed(42)
text = "我爱学习"
print("字符 Token", list(text))
print("空白分词", "I love learning".split())
print("字符数", len(text))

# %% 词表、特殊 Token 与 ID
# 词表把 Token 映射到整数，ID 的数值大小不表示语义远近。保留 PAD=0、UNK=1；未知字符回退到 UNK，避免索引越界。
corpus = "我爱学习你也爱编程"
vocab = {"<PAD>": 0, "<UNK>": 1}
for character in sorted(set(corpus)):
    vocab[character] = len(vocab)
inverse_vocab = {index: token for token, index in vocab.items()}
ids = [vocab.get(token, 1) for token in "我爱学习!"]
print(vocab)
print("Token ID", ids)
print("还原", [inverse_vocab[index] for index in ids])

# %% Embedding 查表
# Embedding 是形状 (词表大小,向量维度) 的可训练矩阵。输入整数 ID，输出对应的行；输入 (B,T) 会变成 (B,T,D)。padding_idx=0 让 PAD 对应行不通过普通 Embedding 梯度更新。
embedding = nn.Embedding(len(vocab), 6, padding_idx=0)
token_ids = torch.tensor([ids])
vectors = embedding(token_ids)
print("输入", token_ids.shape, token_ids.dtype)
print("输出", vectors.shape)
print("PAD 向量", embedding.weight[0])

# %% 变长序列与 Padding
# 同一批次中的序列补到相同长度。attention_mask 的 1 表示有效 Token；PyTorch MultiheadAttention 的 key_padding_mask=True 却表示屏蔽，使用前需要明确约定。
from torch.nn.utils.rnn import pad_sequence
sequences = [torch.tensor([vocab[t] for t in sentence]) for sentence in ["我爱学习", "你也爱"]]
padded = pad_sequence(sequences, batch_first=True, padding_value=0)
attention_mask = padded.ne(0)
print("批次", padded)
print("有效位置", attention_mask)
print("需屏蔽的位置", ~attention_mask)

# %% 语言模型：右移一个 Token
# 自回归语言模型在每个位置预测下一个 Token。训练时把同一序列错开一位得到 x 与 y；输入包含当前 Token，但不能看到未来的目标 Token。
stream = torch.tensor([vocab[token] for token in "我爱学习你也爱学习"])
context_length = 4
inputs = stream[:context_length]
targets = stream[1:context_length + 1]
print("输入", inputs.tolist())
print("目标", targets.tolist())
print(list(zip(inputs.tolist(), targets.tolist())))

# %% 逐 Token 的 logits 与损失
# 模型输出 (B,T,V)，其中 V 是词表大小。CrossEntropyLoss 常将前两个维度展平；PAD 标签可以使用 ignore_index 排除，避免模型把大量补齐位置当作任务。
head = nn.Linear(6, len(vocab))
logits = head(embedding(padded))
labels = padded.clone()
labels[:, :-1] = padded[:, 1:]
labels[:, -1] = 0
loss = nn.functional.cross_entropy(logits.reshape(-1, len(vocab)), labels.reshape(-1), ignore_index=0)
print("logits 与标签", logits.shape, labels.shape)
print("损失", loss.item())

# %% 上下文长度与截断
# 上下文窗口限制一次可使用的历史长度。生成时可以只保留最近一段上下文，但被裁掉的信息无法继续参与本次注意力计算。Padding 不会凭空增加真实信息。
long_ids = torch.arange(12)
max_context = 5
recent_context = long_ids[-max_context:]
print("原序列", long_ids.tolist())
print("当前上下文", recent_context.tolist())
print("批处理形状", recent_context.unsqueeze(0).shape)
