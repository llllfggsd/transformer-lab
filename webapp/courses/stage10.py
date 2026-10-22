from . import course, example, practice, quiz

COURSE = course(10, "进阶内容", "用小规模实验认识现代 Transformer 的位置、归一化、缓存与微调。", "MODERN TRANSFORMERS", [
    example("RoPE：旋转式位置编码", "RoPE 通常作用于 Q/K。这里采用相邻两维成对旋转的约定；旋转保留每对向量的长度，并让点积感受到相对位置。不同实现的维度配对方式可能不同，迁移权重时必须匹配。", '''
        import torch
        from torch import nn
        import math
        torch.set_num_threads(1)
        torch.manual_seed(42)
        def rope(x, positions):
            dimension = x.shape[-1]
            inverse = 10000. ** (-torch.arange(0, dimension, 2).float() / dimension)
            angle = positions.float().unsqueeze(-1) * inverse
            even, odd = x[..., 0::2], x[..., 1::2]
            paired = torch.stack([even * angle.cos() - odd * angle.sin(), even * angle.sin() + odd * angle.cos()], dim=-1)
            return paired.flatten(-2)
        x = torch.randn(3, 8)
        rotated = rope(x, torch.arange(3))
        print("长度保持", torch.allclose(x.norm(dim=-1), rotated.norm(dim=-1), atol=1e-6))
    '''),
    example("RMSNorm", "RMSNorm 按均方根缩放，不像 LayerNorm 那样先减去均值。常见实现再乘可训练的缩放参数。eps 防止分母为零；下面只展示无仿射参数的核心公式。", '''
        def rms_norm(x, eps=1e-6):
            return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
        values = torch.tensor([[1., 2., 3., 4.]])
        print("RMSNorm", rms_norm(values))
        print("LayerNorm", nn.functional.layer_norm(values, (4,)))
    '''),
    example("SwiGLU 前馈网络", "SwiGLU 使用 SiLU(gate(x)) 与 up(x) 逐元素相乘，再由 down 投影回原维度。它的三组投影与普通两层 FFN 不同，比较参数量时需要控制中间维度。", '''
        dimension, hidden = 8, 16
        gate = nn.Linear(dimension, hidden, bias=False)
        up = nn.Linear(dimension, hidden, bias=False)
        down = nn.Linear(hidden, dimension, bias=False)
        tokens = torch.randn(2, 4, dimension)
        output = down(nn.functional.silu(gate(tokens)) * up(tokens))
        print("SwiGLU 输出", output.shape)
    '''),
    example("KV Cache：追加缓存", "生成新 Token 时，可缓存历史 K/V，只计算新位置的 Q/K/V。缓存减少重复投影和历史层计算，但新 Query 仍需读取历史 Key。这里只验证单头、无位置编码的核心等价关系。", '''
        q, k, v = torch.randn(1, 4, 8), torch.randn(1, 4, 8), torch.randn(1, 4, 8)
        cached_k, cached_v = k[:, :3], v[:, :3]
        cached_k = torch.cat([cached_k, k[:, 3:4]], dim=1)
        cached_v = torch.cat([cached_v, v[:, 3:4]], dim=1)
        incremental = (q[:, 3:4] @ cached_k.transpose(-2, -1) / math.sqrt(8)).softmax(-1) @ cached_v
        full_last = (q[:, -1:] @ k.transpose(-2, -1) / math.sqrt(8)).softmax(-1) @ v
        print("缓存与完整计算一致", torch.allclose(incremental, full_last))
        print("缓存元素数", cached_k.numel() + cached_v.numel())
    '''),
    example("长文本与局部注意力", "完整注意力的分数矩阵有 T² 个元素。局部窗口减少允许连接的数量，但下面仍创建稠密矩阵，仅演示可见范围，并不自动实现稀疏加速。FlashAttention 侧重降低存储访问，并没有把所有计算都变成线性复杂度。", '''
        length, window = 8, 3
        query_position = torch.arange(length).unsqueeze(1)
        key_position = torch.arange(length).unsqueeze(0)
        blocked = (key_position > query_position) | (key_position < query_position - window + 1)
        print("局部因果屏蔽", blocked.int())
        print("允许连接数 / 完整分数数量", (~blocked).sum().item(), length ** 2)
    '''),
    example("对称 int8 量化", "用 scale=max(abs(weight))/127 将浮点权重映射到 [-127,127]，再反量化近似原值。量化通常减少存储，但精度会有误差；这里是数值演示，不是高性能整数推理内核。", '''
        weight = torch.tensor([[-1., -0.2, 0., 0.4, 1.]])
        scale = weight.abs().max().clamp_min(1e-8) / 127
        quantized = (weight / scale).round().clamp(-127, 127).to(torch.int8)
        reconstructed = quantized.float() * scale
        print(quantized, scale.item())
        print("最大误差", (weight - reconstructed).abs().max().item())
        print("权重字节", weight.numel() * weight.element_size(), quantized.numel() * quantized.element_size())
    '''),
    example("LoRA：只训练低秩增量", "冻结基础 W，只训练 A 与 B，增量为 (alpha/r)BA。B 初始化为零可让初始输出等于原模型。第一次更新时 A 的梯度可能为零，因为此时 B 为零，这不表示实现失败。", '''
        base = nn.Linear(8, 8, bias=False)
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        rank, alpha = 2, 4.
        a = nn.Parameter(torch.randn(rank, 8) * 0.01)
        b = nn.Parameter(torch.zeros(8, rank))
        features = torch.randn(3, 8)
        original = base(features)
        adapted = original + (features @ a.T @ b.T) * (alpha / rank)
        print("初始输出相同", torch.allclose(original, adapted))
        print("低秩参数 / 原矩阵参数", a.numel() + b.numel(), base.weight.numel())
        adapted.square().mean().backward()
        print("基础权重梯度", base.weight.grad)
    '''),
    example("指令微调：只对回答计损失", "指令微调学习按指令输出回答。一个常见做法是将提示部分标签设为 -100，只对回答 Token 计算交叉熵。实际语言模型还要正确处理因果右移与聊天模板；这里的 logits 已与标签位置对齐。", '''
        logits = torch.randn(1, 5, 10, requires_grad=True)
        labels = torch.tensor([[-100, -100, 3, 4, 5]])
        loss = nn.functional.cross_entropy(logits.reshape(-1, 10), labels.reshape(-1), ignore_index=-100)
        loss.backward()
        print("回答损失", loss.item())
        print("提示位置梯度为零", logits.grad[:, :2].abs().sum().item())
    '''),
    example("偏好对齐：一个 DPO 损失例子", "DPO 使用同一提示下的优选/拒绝回答，比较策略与冻结参考模型的回答对数概率差。这里给定已经汇总的回答 log probability，只演示成对损失；完整训练还需要正确掩码、序列求和、参考模型和偏好数据。", '''
        policy_chosen = torch.tensor([-2.0], requires_grad=True)
        policy_rejected = torch.tensor([-4.0], requires_grad=True)
        reference_chosen = torch.tensor([-3.0])
        reference_rejected = torch.tensor([-3.5])
        beta = 0.1
        margin = (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)
        preference_loss = -nn.functional.logsigmoid(beta * margin).mean()
        preference_loss.backward()
        print("DPO 损失", preference_loss.item())
        print("优选/拒绝梯度", policy_chosen.grad, policy_rejected.grad)
    '''),
], [
    practice("RMSNorm 核心", "沿最后一维实现无仿射参数 RMSNorm：x / sqrt(mean(x²)+eps)。保留形状，默认 eps=1e-6。", 'def normalize_rms(x, eps=1e-6):\n    raise NotImplementedError', '''
        def normalize_rms(x, eps=1e-6):
            return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    ''', '''
        x = torch.tensor([[1., 2., 3.], [0., 0., 0.]])
        torch.testing.assert_close(normalize_rms(x), x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6))
        assert torch.isfinite(normalize_rms(torch.zeros(2, 4))).all()
    '''),
    practice("局部因果掩码", "返回 (length,length) 布尔掩码。Query i 只看 max(0,i-window+1) 到 i，其余为 True。length 与 window 为正整数。", 'def local_causal_mask(length, window):\n    raise NotImplementedError', '''
        def local_causal_mask(length, window):
            i = torch.arange(length).unsqueeze(1)
            j = torch.arange(length).unsqueeze(0)
            return (j > i) | (j < i - window + 1)
    ''', '''
        torch.testing.assert_close(local_causal_mask(4, 2), torch.tensor([[False, True, True, True], [False, False, True, True], [True, False, False, True], [True, True, False, False]]))
        assert (~local_causal_mask(3, 1)).sum() == 3
    '''),
    practice("对称量化", "返回 (quantized,scale)，quantized 为 int8，scale 为正的标量 Tensor。scale=max(max(abs(x)),1e-8)/127，先四舍五入再裁到 [-127,127]。", 'def quantize_int8(x):\n    raise NotImplementedError', '''
        def quantize_int8(x):
            scale = x.abs().max().clamp_min(1e-8) / 127
            return (x / scale).round().clamp(-127, 127).to(torch.int8), scale
    ''', '''
        x = torch.tensor([-1., 0., 1.])
        q, scale = quantize_int8(x)
        assert q.dtype == torch.int8 and scale > 0
        torch.testing.assert_close(q, torch.tensor([-127, 0, 127], dtype=torch.int8))
        torch.testing.assert_close(q.float() * scale, x)
        q, scale = quantize_int8(torch.zeros(4))
        assert (q == 0).all() and scale > 0
    '''),
], [
    quiz("KV Cache 主要保存什么？", ["历史 Key 和 Value", "训练集标签", "所有优化器梯度"], 0),
    quiz("LoRA 通常训练哪些参数？", ["必须更新全部原权重", "低秩增量参数", "只改分词器"], 1),
    quiz("本例稠密局部 Mask 是否自动带来稀疏计算加速？", ["一定", "把 True 改成 False 就可以", "不会，需要合适的实现或内核"], 2),
], requirements=["torch"], sources=["https://arxiv.org/abs/2104.09864", "https://arxiv.org/abs/2106.09685", "https://arxiv.org/abs/2305.18290"])
