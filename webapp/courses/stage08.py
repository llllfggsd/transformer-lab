from . import course, example, practice, quiz

COURSE = course(8, "常见模型与训练方式", "用无需下载权重的微型 BERT、GPT 和 T5 区分结构与训练目标。", "MODEL FAMILIES", [
    example("Encoder-only：微型 BERT", "BERT 使用双向 Encoder，适合表示学习、分类和掩码预测。这里从配置创建真实 BertModel，但权重随机，不能据此判断文本含义。所有配置都很小，不会下载模型。首次导入依赖可能较慢。", '''
        import torch
        from torch import nn
        from transformers import BertConfig, BertModel, GPT2Config, GPT2LMHeadModel, T5Config, T5ForConditionalGeneration
        torch.set_num_threads(1)
        torch.manual_seed(42)
        bert_config = BertConfig(vocab_size=24, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, max_position_embeddings=32)
        bert = BertModel(bert_config).eval()
        token_ids = torch.tensor([[3, 4, 5, 6]])
        with torch.no_grad():
            encoded = bert(token_ids).last_hidden_state
        print("BERT 隐状态", encoded.shape)
    '''),
    example("Decoder-only：微型 GPT", "GPT 类模型使用因果注意力做下一个 Token 预测。GPT2LMHeadModel 在 labels=input_ids 时会在内部错开预测与标签；不要再手动右移一次。下面的模型同样是随机初始化。", '''
        gpt_config = GPT2Config(vocab_size=24, n_positions=16, n_embd=16, n_layer=1, n_head=2, bos_token_id=1, eos_token_id=2)
        gpt = GPT2LMHeadModel(gpt_config).eval()
        with torch.no_grad():
            result = gpt(token_ids, labels=token_ids, use_cache=False)
        print("GPT logits", result.logits.shape)
        print("下一个 Token 损失", result.loss.item())
    '''),
    example("Encoder–Decoder：微型 T5", "T5 的 Encoder 读取输入，Decoder 自回归生成目标并通过交叉注意力读取输入表示。给 labels 时库会创建右移后的 Decoder 输入。原始 Transformer 和 T5 都属于这一家族，但细节不同。", '''
        t5_config = T5Config(vocab_size=24, d_model=16, d_ff=32, d_kv=8, num_heads=2, num_layers=1, num_decoder_layers=1, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
        t5 = T5ForConditionalGeneration(t5_config).eval()
        labels = torch.tensor([[7, 8, 1]])
        with torch.no_grad():
            result = t5(input_ids=token_ids, attention_mask=torch.ones_like(token_ids), labels=labels, use_cache=False)
        print("T5 目标 logits", result.logits.shape)
        print("条件生成损失", result.loss.item())
    '''),
    example("预训练任务：掩码预测", "预训练先从大量数据学习通用参数。BERT 的掩码语言建模仅在选定位置计算损失；这里只演示把一个位置替换成特殊 ID，不实现完整的随机替换策略。双向模型可以使用左右上下文。", '''
        original = token_ids.clone()
        masked = original.clone()
        masked[:, 1] = 23
        prediction_head = nn.Linear(16, 24)
        hidden = bert(masked).last_hidden_state
        masked_loss = nn.functional.cross_entropy(prediction_head(hidden[:, 1]), original[:, 1])
        print("掩码输入", masked)
        print("被遮住位置的损失", masked_loss.item())
    '''),
    example("微调与冻结参数", "微调是在已有模型参数上继续训练。下面只演示冻结 Encoder、训练分类头的机制；由于 Encoder 尚未预训练，这不是可用的情感分类器。真实迁移应加载经过训练的权重。", '''
        for parameter in bert.parameters():
            parameter.requires_grad_(False)
        classifier = nn.Linear(16, 2)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
        with torch.no_grad():
            sentence_vector = bert(token_ids).last_hidden_state.mean(dim=1)
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(classifier(sentence_vector), torch.tensor([1]))
        loss.backward()
        optimizer.step()
        print("可训练 Encoder 参数", sum(p.numel() for p in bert.parameters() if p.requires_grad))
        print("可训练分类头参数", sum(p.numel() for p in classifier.parameters()))
    '''),
    example("温度、Top-k 与采样", "温度 T>0 用 logits/T 改变概率分布；T 越低越集中。Top-k 只保留分数最高的 k 个候选，再归一化并采样。采样控制选择方式，不会让未经训练的模型获得知识。", '''
        logits = torch.tensor([1., 2., 3., 4.])
        for temperature in [0.5, 1., 2.]:
            print(temperature, (logits / temperature).softmax(dim=-1))
        threshold = torch.topk(logits, 2).values[-1]
        filtered = logits.masked_fill(logits < threshold, float("-inf"))
        sampled = torch.multinomial(filtered.softmax(dim=-1), num_samples=1)
        print("Top-k 采样 ID", sampled.item())
    '''),
    example("自回归生成与 LLM 的关系", "每次用已有 Token 预测一个新 Token，再把它接回上下文。Decoder-only 是常见 LLM 架构，但 Transformer 本身不等于大语言模型：参数规模、数据与训练过程同样关键。下面只观察随机小模型的生成流程。", '''
        generated = token_ids.clone()
        with torch.no_grad():
            for step in range(4):
                logits = gpt(generated[:, -16:], use_cache=False).logits[:, -1]
                next_id = logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_id], dim=1)
        print("生成的 ID", generated.tolist())
        print("小模型参数数目", sum(p.numel() for p in gpt.parameters()))
    '''),
], [
    practice("温度概率", "temperature_probs(logits, temperature) 沿最后一轴返回 Softmax(logits/temperature)。温度 <=0 时抛出 ValueError。", 'def temperature_probs(logits, temperature):\n    raise NotImplementedError', '''
        def temperature_probs(logits, temperature):
            if temperature <= 0:
                raise ValueError("温度必须为正")
            return torch.softmax(logits / temperature, dim=-1)
    ''', '''
        scores = torch.tensor([[1., 2., 3.]])
        torch.testing.assert_close(temperature_probs(scores, 2.), torch.softmax(scores / 2, -1))
        try:
            temperature_probs(scores, 0.)
        except ValueError:
            pass
        else:
            raise AssertionError("温度为零应报错")
    '''),
    practice("Top-k 过滤", "logits 是一维且分数无并列，1<=k<=词表大小。返回同形状张量，仅保留最高 k 项原值，其余为 -inf，不修改输入。", 'def top_k_logits(logits, k):\n    raise NotImplementedError', '''
        def top_k_logits(logits, k):
            threshold = torch.topk(logits, k).values[-1]
            return logits.masked_fill(logits < threshold, float("-inf"))
    ''', '''
        data = torch.tensor([1., 4., 2., 3.])
        result = top_k_logits(data, 2)
        assert torch.isneginf(result[[0, 2]]).all()
        torch.testing.assert_close(result[[1, 3]], torch.tensor([4., 3.]))
        torch.testing.assert_close(data, torch.tensor([1., 4., 2., 3.]))
        torch.testing.assert_close(top_k_logits(data, 4), data)
    '''),
    practice("只训练指定的分类头", "freeze_backbone(backbone, head) 让 backbone 的全部参数停止求导，让 head 的参数可求导；原地修改，不必返回值。", 'def freeze_backbone(backbone, head):\n    raise NotImplementedError', '''
        def freeze_backbone(backbone, head):
            for parameter in backbone.parameters():
                parameter.requires_grad_(False)
            for parameter in head.parameters():
                parameter.requires_grad_(True)
    ''', '''
        base = nn.Linear(4, 4)
        head = nn.Linear(4, 2)
        for p in head.parameters():
            p.requires_grad_(False)
        freeze_backbone(base, head)
        assert all(not p.requires_grad for p in base.parameters())
        assert all(p.requires_grad for p in head.parameters())
    '''),
], [
    quiz("从配置创建 GPT 模型意味着什么？", ["已经有语言能力", "随机权重，需要训练", "自动下载完整预训练权重"], 1),
    quiz("哪种结构包含读取 Encoder memory 的交叉注意力？", ["Encoder–Decoder", "所有 Decoder-only", "只有 Embedding"], 0),
    quiz("降低正温度通常会怎样改变分布？", ["必然更随机", "删除所有概率", "使高分候选更集中"], 2),
], requirements=["torch", "transformers"], sources=["https://huggingface.co/docs/transformers/model_doc/bert", "https://huggingface.co/docs/transformers/model_doc/gpt2", "https://huggingface.co/docs/transformers/model_doc/t5"])
COURSE["lessons"][0]["timeout"] = 60
