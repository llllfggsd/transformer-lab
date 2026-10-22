from . import course, example, practice, quiz

COURSE = course(4, "深度学习与 PyTorch", "用自动求导、模型和训练循环完成一个小型分类器。", "PYTORCH", [
    example("张量与自动求导", "Tensor 是 PyTorch 的数值容器。requires_grad=True 让系统记录计算图；backward() 把标量损失的梯度累加到叶子张量的 .grad。此课程统一使用 CPU，固定种子便于重现。", '''
        import torch
        from torch import nn
        torch.set_num_threads(1)
        torch.manual_seed(42)
        x = torch.tensor([2., 3.], requires_grad=True)
        loss = (x ** 2).sum()
        loss.backward()
        print("损失", loss.item(), "梯度", x.grad)
    '''),
    example("前向传播、反向传播与清零", "前向计算 y=wx+b，反向计算参数对损失的导数。梯度默认累加，所以重复 backward 前要清零。优化器的 zero_grad() 也是解决这个问题。", '''
        weight = torch.tensor(3., requires_grad=True)
        bias = torch.tensor(1., requires_grad=True)
        prediction = weight * 2 + bias
        loss = (prediction - 5) ** 2
        loss.backward()
        print("预测与损失", prediction.item(), loss.item())
        print("参数梯度", weight.grad.item(), bias.grad.item())
        weight.grad.zero_()
        bias.grad.zero_()
    '''),
    example("神经元、激活函数与多层网络", "线性层执行 xWᵀ+b。多层线性层中间加入非线性激活，才能表示比单个线性映射更复杂的关系。ReLU 会把负数变成零；GELU 更平滑。", '''
        values = torch.tensor([-2., -1., 0., 1., 2.])
        print("ReLU", torch.relu(values))
        print("GELU", nn.functional.gelu(values))
        network = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
        print("批量输出", network(torch.ones(4, 2)).shape)
        print("参数数量", sum(p.numel() for p in network.parameters()))
    '''),
    example("用类搭建分类模型", "继承 nn.Module 并在 __init__ 注册层，在 forward 定义数据流。分类器输出原始 logits，CrossEntropyLoss 内部完成归一化，不需要先做 Softmax。", '''
        class SmallClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(nn.Linear(2, 12), nn.Tanh(), nn.Linear(12, 2))
            def forward(self, features):
                return self.layers(features)
        model = SmallClassifier()
        print(model)
        print(model(torch.zeros(3, 2)).shape)
    '''),
    example("Batch、Epoch 与 DataLoader", "Batch 是一次更新使用的样本数，Epoch 是遍历完整训练集一次。固定训练/验证划分；每轮打乱训练顺序有助于减少固定顺序的影响。", '''
        from torch.utils.data import TensorDataset, DataLoader
        features = torch.randn(160, 2)
        labels = (features[:, 0] + features[:, 1] > 0).long()
        train_data = TensorDataset(features[:128], labels[:128])
        loader = DataLoader(train_data, batch_size=32, shuffle=True)
        validation_x, validation_y = features[128:], labels[128:]
        print("每轮批次数", len(loader))
        batch_x, batch_y = next(iter(loader))
        print(batch_x.shape, batch_y.shape)
    '''),
    example("完成训练循环", "训练一步依次执行清梯度、前向、损失、反向、优化器更新。model.train() 设置训练模式；学习率决定步长，不能与 Epoch 数混为一谈。", '''
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
    '''),
    example("验证与推理", "model.eval() 切换 Dropout、BatchNorm 等层的行为；torch.no_grad() 关闭求导记录。两者作用不同，推理时通常一起用。这里的准确率只来自小型合成验证集。", '''
        model.eval()
        with torch.no_grad():
            logits = model(validation_x)
            predicted = logits.argmax(dim=-1)
            accuracy = (predicted == validation_y).float().mean()
        print("验证准确率", accuracy.item())
        print("单个样本的概率", torch.softmax(logits[0], dim=-1))
    '''),
    example("保存与重新加载参数", "state_dict 保存已注册参数。加载时需要建立相同结构，再读入参数；仅加载权重后用 eval 模式推理。此处文件属于当前账号的工作目录。", '''
        torch.save(model.state_dict(), "stage04_classifier.pt")
        restored = SmallClassifier()
        restored.load_state_dict(torch.load("stage04_classifier.pt", weights_only=True))
        restored.eval()
        with torch.no_grad():
            print("加载结果一致", torch.allclose(model(validation_x), restored(validation_x)))
    '''),
], [
    practice("自动求导的平方梯度", "实现 square_gradient(values)：输入一维浮点张量，返回 sum(values²) 对各元素的梯度，不修改输入的 requires_grad 或已有 grad。", 'def square_gradient(values):\n    raise NotImplementedError', '''
        def square_gradient(values):
            leaf = values.detach().clone().requires_grad_(True)
            (leaf ** 2).sum().backward()
            return leaf.grad
    ''', '''
        values = torch.tensor([-2., 0., 3.])
        torch.testing.assert_close(square_gradient(values), torch.tensor([-4., 0., 6.]))
        assert not values.requires_grad and values.grad is None
    '''),
    practice("可训练的线性层", "实现继承 nn.Module 的 LinearRegressor(input_dim)。注册一个 nn.Linear(input_dim,1)，forward 返回 (B,1)。参数必须可被优化器获取。", 'class LinearRegressor(nn.Module):\n    def __init__(self, input_dim):\n        super().__init__()\n        raise NotImplementedError\n    def forward(self, x):\n        raise NotImplementedError', '''
        class LinearRegressor(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.linear = nn.Linear(input_dim, 1)
            def forward(self, x):
                return self.linear(x)
    ''', '''
        model_under_test = LinearRegressor(3)
        assert isinstance(model_under_test, nn.Module)
        output = model_under_test(torch.ones(4, 3))
        assert output.shape == (4, 1)
        assert sum(p.numel() for p in model_under_test.parameters()) == 4
        output.sum().backward()
        assert all(p.grad is not None for p in model_under_test.parameters())
    '''),
    practice("训练一步", "实现 train_step(model, optimizer, x, y)：使用 MSE，清梯度、反向并更新参数，返回本次更新前的损失浮点数。", 'def train_step(model, optimizer, x, y):\n    raise NotImplementedError', '''
        def train_step(model, optimizer, x, y):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()
            return loss.item()
    ''', '''
        layer = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            layer.weight.fill_(0.)
        opt = torch.optim.SGD(layer.parameters(), lr=0.1)
        value = train_step(layer, opt, torch.ones(2, 1), torch.ones(2, 1))
        assert abs(value - 1.) < 1e-6
        torch.testing.assert_close(layer.weight, torch.tensor([[0.2]]))
        train_step(layer, opt, torch.ones(2, 1), torch.ones(2, 1))
        torch.testing.assert_close(layer.weight, torch.tensor([[0.36]]))
    '''),
], [
    quiz("训练一步中哪项在 backward 之后？", ["optimizer.step()", "定义训练集", "optimizer.zero_grad() 必须只在之后"], 0),
    quiz("CrossEntropyLoss 通常接收什么？", ["先手动 Softmax 的概率", "原始 logits 和整数类别标签", "预测类别的 argmax"], 1),
    quiz("eval() 和 no_grad() 的关系是什么？", ["完全相同", "都修改模型权重", "分别控制层行为和梯度记录"], 2),
], requirements=["torch"], sources=["https://docs.pytorch.org/tutorials/beginner/basics/intro.html"])
