from . import course, example, practice, quiz

COURSE = course(3, "机器学习基础", "用可观察的损失、数据划分和参数更新理解训练与泛化。", "MACHINE LEARNING", [
    example("模型、参数与推理", "模型是带参数的计算规则。线性模型 y=xw+b 中，w 和 b 是参数；给定参数计算预测叫推理，通过数据调整参数叫训练。", '''
        import numpy as np
        features = np.array([[-2.], [-1.], [0.], [1.], [2.], [3.]])
        targets = 2 * features + 1
        weight, bias = 0.5, 0.
        predictions = features * weight + bias
        print("预测", predictions.ravel())
        print("目标", targets.ravel())
    '''),
    example("训练、验证与测试划分", "训练集更新参数，验证集选择学习率等设置，测试集只做最终评估。时间序列应按时间切分；这里的独立样本使用固定随机种子划分，保证可重复。", '''
        rng = np.random.default_rng(42)
        indices = rng.permutation(30)
        train_ids = indices[:18]
        validation_ids = indices[18:24]
        test_ids = indices[24:]
        print(len(train_ids), len(validation_ids), len(test_ids))
        print("无交集", len(set(train_ids) & set(test_ids)) == 0)
    '''),
    example("损失函数：均方误差", "回归任务预测连续数值。MSE 对预测误差平方后求平均；损失是一个标量，用于衡量当前参数在一批数据上的表现。", '''
        errors = predictions - targets
        mse = np.mean(errors ** 2)
        print("均方误差", mse)
        print("完美预测的损失", np.mean((targets - targets) ** 2))
    '''),
    example("优化器：手写 SGD", "SGD 是按梯度更新参数的规则。对均方误差，dL/dw=2*mean((pred-y)*x)，dL/db=2*mean(pred-y)。每轮更新后观察损失是否下降。", '''
        weight, bias = 0., 0.
        for epoch in range(30):
            predicted = features * weight + bias
            error = predicted - targets
            weight -= 0.05 * 2 * np.mean(error * features)
            bias -= 0.05 * 2 * np.mean(error)
        print("训练参数", round(weight, 3), round(bias, 3))
        print("训练后损失", np.mean((features * weight + bias - targets) ** 2))
    '''),
    example("分类、概率与交叉熵", "分类预测离散类别，回归预测连续值。二分类可用 Sigmoid 把分数转成正类概率，再使用二元交叉熵。阈值 0.5 是一种决策规则，不是概率本身。", '''
        scores = np.array([-2., -0.5, 0.5, 2.])
        labels = np.array([0., 0., 1., 1.])
        probabilities = 1 / (1 + np.exp(-scores))
        clipped = np.clip(probabilities, 1e-8, 1 - 1e-8)
        cross_entropy = -np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
        print("概率", probabilities)
        print("交叉熵", cross_entropy, "准确率", np.mean((probabilities >= 0.5) == labels))
    '''),
    example("欠拟合、过拟合与泛化", "容量太小可能欠拟合；高次多项式可能把噪声一起记住。比较训练与独立验证误差，比只看训练误差更可靠。这个小例子只展示机制，单次切分不能证明哪个模型普遍更优。", '''
        rng = np.random.default_rng(7)
        train_x = np.linspace(-1, 1, 9)
        train_y = np.sin(3 * train_x) + rng.normal(0, 0.15, 9)
        validation_x = np.linspace(-0.95, 0.95, 40)
        validation_y = np.sin(3 * validation_x)
        for degree in [1, 3, 8]:
            coefficients = np.polyfit(train_x, train_y, degree)
            train_error = np.mean((np.polyval(coefficients, train_x) - train_y) ** 2)
            validation_error = np.mean((np.polyval(coefficients, validation_x) - validation_y) ** 2)
            print(degree, round(train_error, 4), round(validation_error, 4))
    '''),
    example("预处理也要避免泄漏", "标准化的均值和标准差必须只从训练集计算，再应用于验证和测试集。把验证数据加入统计量会泄漏信息，即使没有使用标签也不合适。", '''
        training_values = np.array([[1., 10.], [3., 20.], [5., 30.]])
        validation_values = np.array([[7., 40.]])
        mean = training_values.mean(axis=0)
        std = training_values.std(axis=0) + 1e-8
        print("训练标准化", (training_values - mean) / std)
        print("验证标准化", (validation_values - mean) / std)
    '''),
], [
    practice("均方误差", "返回同形状 NumPy 数组 prediction 和 target 的均方误差标量。", 'def mse_loss(prediction, target):\n    raise NotImplementedError', '''
        def mse_loss(prediction, target):
            return np.mean((prediction - target) ** 2)
    ''', '''
        assert np.isclose(mse_loss(np.array([1., 3.]), np.array([2., 1.])), 2.5)
        assert mse_loss(np.ones((2, 3)), np.ones((2, 3))) == 0
    '''),
    practice("分类准确率", "logits 形状 (B,C)，labels 形状 (B,)。取最后一轴 argmax 作为预测，返回准确率。", 'def classification_accuracy(logits, labels):\n    raise NotImplementedError', '''
        def classification_accuracy(logits, labels):
            return np.mean(np.argmax(logits, axis=-1) == labels)
    ''', '''
        scores = np.array([[1., 3., 0.], [4., 1., 2.], [0., 1., 5.]])
        assert np.isclose(classification_accuracy(scores, np.array([1, 0, 0])), 2 / 3)
        assert classification_accuracy(np.array([[0., 1.]]), np.array([1])) == 1
    '''),
    practice("训练统计量标准化", "用 train 的逐列均值和标准差标准化 values；标准差小于 1e-8 时使用 1，避免除零。返回与 values 同形状的数组。", 'def standardize(train, values):\n    raise NotImplementedError', '''
        def standardize(train, values):
            mean = train.mean(axis=0)
            std = train.std(axis=0)
            std = np.where(std < 1e-8, 1., std)
            return (values - mean) / std
    ''', '''
        np.testing.assert_allclose(standardize(np.array([[1., 2.], [3., 2.]]), np.array([[5., 2.]])), [[3., 0.]])
        assert standardize(np.ones((3, 2)), np.ones((2, 2))).shape == (2, 2)
    '''),
], [
    quiz("哪个数据集用于选择学习率？", ["测试集", "验证集", "把全部数据合并"], 1),
    quiz("训练误差很低而验证误差很高，通常提示什么？", ["可能过拟合", "一定已经泛化", "必须继续降低训练误差"], 0),
    quiz("标准化参数应从哪里计算？", ["测试集", "全部数据", "仅训练集"], 2),
])
