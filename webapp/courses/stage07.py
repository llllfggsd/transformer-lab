from . import course, example, practice, quiz

COURSE = course(7, "Transformer 的完整结构", "把位置、注意力、前馈网络、残差和归一化串成完整的数据流。", "TRANSFORMER BLOCKS", [
    example("位置编码：给顺序一个坐标", "不加入位置信息时，普通自注意力本身缺乏位置标签。正弦位置编码使用不同频率的 sin/cos，输出 (T,D)，再广播到 (B,T,D)。这里 D 为偶数。", '''
        import torch
        from torch import nn
        import math
        torch.set_num_threads(1)
        torch.manual_seed(42)
        def sinusoidal_positions(length, dimension):
            positions = torch.arange(length).float().unsqueeze(1)
            frequency = torch.exp(torch.arange(0, dimension, 2).float() * (-math.log(10000.) / dimension))
            encoding = torch.zeros(length, dimension)
            encoding[:, 0::2] = torch.sin(positions * frequency)
            encoding[:, 1::2] = torch.cos(positions * frequency)
            return encoding
        positions = sinusoidal_positions(5, 8)
        print(positions.shape, positions[0])
    '''),
    example("前馈网络 FFN", "FFN 在每个 Token 上独立应用相同的两层映射，先扩展维度、激活，再压回 D。它混合特征维度，注意力则让不同 Token 交换信息。", '''
        tokens = torch.randn(2, 5, 8) + positions.unsqueeze(0)
        ffn = nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Linear(32, 8))
        print("FFN 输入与输出", tokens.shape, ffn(tokens).shape)
    '''),
    example("残差与 LayerNorm", "残差连接把子层输出加回输入，要求两者形状相同。LayerNorm(8) 沿最后 8 个特征归一化。原论文采用 Post-Norm，很多现代模型采用 Pre-Norm，两者计算顺序不同。", '''
        norm = nn.LayerNorm(8)
        pre_norm_output = tokens + ffn(norm(tokens))
        post_norm_output = norm(tokens + ffn(tokens))
        print(pre_norm_output.shape, post_norm_output.shape)
        print("归一化后的均值", norm(tokens).mean(dim=-1))
    '''),
    example("亲手串联一个 Pre-Norm 模块", "顺序为 x + Attention(LN(x))，然后 x + FFN(LN(x))。注意力可以接收因果掩码，同一个模块可用不同 Mask 控制可见范围。", '''
        class TransformerBlock(nn.Module):
            def __init__(self, dimension=8, heads=2):
                super().__init__()
                self.norm1 = nn.LayerNorm(dimension)
                self.norm2 = nn.LayerNorm(dimension)
                self.attn = nn.MultiheadAttention(dimension, heads, dropout=0., batch_first=True)
                self.ffn = nn.Sequential(nn.Linear(dimension, 4 * dimension), nn.GELU(), nn.Linear(4 * dimension, dimension))
            def forward(self, x, mask=None):
                normalized = self.norm1(x)
                x = x + self.attn(normalized, normalized, normalized, attn_mask=mask, need_weights=False)[0]
                return x + self.ffn(self.norm2(x))
        block = TransformerBlock()
        print("模块输出", block(tokens).shape)
    '''),
    example("Encoder 编码器", "编码器使用双向自注意力，把源序列转成上下文表示。Padding Key 需要被屏蔽。下面使用 PyTorch 的一层 Encoder；D=8，H=2，FFN 隐层=32。", '''
        encoder_layer = nn.TransformerEncoderLayer(8, 2, dim_feedforward=32, dropout=0., batch_first=True)
        encoder = nn.TransformerEncoder(encoder_layer, num_layers=1, enable_nested_tensor=False)
        source_padding = torch.tensor([[False] * 5, [False, False, False, True, True]])
        memory = encoder(tokens, src_key_padding_mask=source_padding)
        print("Encoder memory", memory.shape)
    '''),
    example("Decoder 与交叉注意力", "原始 Decoder 先做因果自注意力，再用目标序列作 Query、Encoder memory 作 Key/Value 做交叉注意力。GPT 类 Decoder-only 通常不包含这一步交叉注意力。", '''
        target = torch.randn(2, 3, 8) + sinusoidal_positions(3, 8)
        target_mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
        decoder_layer = nn.TransformerDecoderLayer(8, 2, dim_feedforward=32, dropout=0., batch_first=True)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        decoded = decoder(target, memory, tgt_mask=target_mask, memory_key_padding_mask=source_padding)
        print("Decoder 输出", decoded.shape)
        cross_attention = nn.MultiheadAttention(8, 2, batch_first=True)
        print("交叉权重", cross_attention(target, memory, memory)[1].shape)
    '''),
    example("完整 Encoder–Decoder 数据流", "源 ID→Embedding→位置→Encoder memory；右移后的目标 ID→Embedding→位置→Decoder→词表 logits。nn.Transformer 不包含词表 Embedding 与最终分类头，需要自行添加。", '''
        source_ids = torch.tensor([[2, 3, 4, 5], [4, 3, 2, 1]])
        target_ids = torch.tensor([[1, 6, 7], [1, 5, 8]])
        embedding = nn.Embedding(12, 8)
        transformer = nn.Transformer(d_model=8, nhead=2, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=32, dropout=0., batch_first=True)
        source = embedding(source_ids) + sinusoidal_positions(4, 8)
        target = embedding(target_ids) + sinusoidal_positions(3, 8)
        hidden = transformer(source, target, tgt_mask=target_mask)
        logits = nn.Linear(8, 12)(hidden)
        print("源/目标/隐状态/logits", source.shape, target.shape, hidden.shape, logits.shape)
    '''),
], [
    practice("位置编码的第一行", "返回 (length,dimension) 正弦位置编码，dimension 为正偶数，使用公式 pos/10000^(2i/d)。dtype 为 float32。", 'def position_encoding(length, dimension):\n    raise NotImplementedError', '''
        def position_encoding(length, dimension):
            pos = torch.arange(length).float().unsqueeze(1)
            freq = torch.exp(torch.arange(0, dimension, 2).float() * (-math.log(10000.) / dimension))
            result = torch.zeros(length, dimension)
            result[:, 0::2] = torch.sin(pos * freq)
            result[:, 1::2] = torch.cos(pos * freq)
            return result
    ''', '''
        result = position_encoding(3, 4)
        assert result.shape == (3, 4)
        torch.testing.assert_close(result[0], torch.tensor([0., 1., 0., 1.]))
        torch.testing.assert_close(result[1], torch.tensor([math.sin(1), math.cos(1), math.sin(0.01), math.cos(0.01)]))
    '''),
    practice("残差与归一化", "返回 LayerNorm(x + branch)，沿最后一维归一化，eps=1e-5，不使用可训练仿射参数。x 与 branch 同形状。", 'def residual_layer_norm(x, branch):\n    raise NotImplementedError', '''
        def residual_layer_norm(x, branch):
            return nn.functional.layer_norm(x + branch, (x.shape[-1],), eps=1e-5)
    ''', '''
        x = torch.tensor([[[1., 2., 3., 4.]]])
        b = torch.tensor([[[2., 0., 1., 0.]]])
        torch.testing.assert_close(residual_layer_norm(x, b), nn.functional.layer_norm(x + b, (4,)))
        torch.testing.assert_close(residual_layer_norm(torch.ones(2, 3), torch.ones(2, 3)), torch.zeros(2, 3))
    '''),
    practice("逐 Token 前馈网络", "build_ffn(dimension,hidden) 返回 nn.Sequential：Linear(D,H)、GELU、Linear(H,D)。应保持任意前导维度。", 'def build_ffn(dimension, hidden):\n    raise NotImplementedError', '''
        def build_ffn(dimension, hidden):
            return nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(), nn.Linear(hidden, dimension))
    ''', '''
        f = build_ffn(6, 18)
        assert isinstance(f, nn.Sequential) and isinstance(f[1], nn.GELU)
        assert f(torch.randn(2, 4, 6)).shape == (2, 4, 6)
        assert sum(p.numel() for p in f.parameters()) == 240
    '''),
], [
    quiz("交叉注意力的 Key/Value 通常来自哪里？", ["只有未来目标", "Encoder memory", "损失函数"], 1),
    quiz("FFN 主要在哪个维度做变换？", ["每个 Token 的特征维度", "只交换 Batch", "只改变序列长度"], 0),
    quiz("nn.Transformer 是否自动包含 Token Embedding 和词表输出头？", ["包含全部", "只包含词表输出头", "两者都需要自行添加"], 2),
], requirements=["torch"], sources=["https://arxiv.org/abs/1706.03762", "https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html"])
