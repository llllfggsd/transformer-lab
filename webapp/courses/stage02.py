from . import course, example, practice, quiz

COURSE = course(2, "必要的数学基础", "从向量和矩阵，到 Softmax、链式法则与梯度下降。", "MATH FOUNDATIONS", [
    example("向量、矩阵与维度", "向量可表示一个样本，矩阵可把多个样本排列成行。矩阵乘法 (B, D) @ (D, H) 得到 (B, H)。这里 B=2、D=3、H=2；请手算左上角，再检查形状。", '''
        import numpy as np
        x = np.array([[1., 2., 3.], [4., 5., 6.]])
        w = np.array([[1., 0.], [0., 1.], [1., 1.]])
        print(x @ w)
        print(x.shape, w.shape, (x @ w).shape)
        print("点积", np.dot(x[0], w[:, 0]))
    '''),
    example("矩阵转置与广播", "x.T 交换行列；每行加同一个偏置向量使用广播。keepdims=True 保留汇总轴，(B,1) 才能与 (B,D) 逐行相减。", '''
        bias = np.array([0.5, -0.5])
        y = x @ w + bias
        row_mean = y.mean(axis=1, keepdims=True)
        print("转置", x.T.shape)
        print("每行居中", y - row_mean)
        print("平均值形状", row_mean.shape)
    '''),
    example("导数、偏导数与数值近似", "f(x)=x² 的导数为 2x。有限差分只是近似，用来检查解析推导；h 太大误差明显，太小会有浮点消减误差。偏导表示固定其他变量时的变化率。", '''
        def f(value):
            return value ** 2
        point = 3.0
        h = 1e-5
        numerical = (f(point + h) - f(point - h)) / (2 * h)
        print("数值导数与解析导数", numerical, 2 * point)
        a, b = 2., 3.
        print("g(a,b)=a*b+a*a 的偏导", b + 2 * a, a)
    '''),
    example("链式法则", "设 z=wx+b、L=z²，链式法则给出 dL/dw=(dL/dz)(dz/dw)=2zx。反向传播就是在计算图上复用这种局部导数。", '''
        feature, weight, bias = 2., 3., 1.
        z = weight * feature + bias
        loss = z ** 2
        dloss_dz = 2 * z
        dloss_dw = dloss_dz * feature
        dloss_db = dloss_dz
        print("损失", loss, "梯度", dloss_dw, dloss_db)
    '''),
    example("指数、对数与稳定 Softmax", "Softmax 把分数变成和为 1 的概率。每个分数先减去最大值可避免 exp 溢出，概率保持不变。axis=-1 表示沿最后一个维度归一化。", '''
        def softmax(values):
            shifted = values - np.max(values, axis=-1, keepdims=True)
            weights = np.exp(shifted)
            return weights / weights.sum(axis=-1, keepdims=True)
        logits = np.array([[1000., 1001., 1002.], [1., 1., 1.]])
        probabilities = softmax(logits)
        print(probabilities)
        print("每行之和", probabilities.sum(axis=-1))
        print("指数与对数互逆", np.log(np.exp(2.)))
    '''),
    example("概率、条件概率与期望", "离散分布的概率非负且总和为 1。期望是按概率加权的平均值；条件概率 P(A|B)=P(A∩B)/P(B)，要求 P(B)>0。", '''
        outcomes = np.array([0., 1., 2.])
        probabilities = np.array([0.2, 0.5, 0.3])
        expectation = (outcomes * probabilities).sum()
        joint_probability, condition_probability = 0.15, 0.30
        print("期望", expectation)
        print("条件概率", joint_probability / condition_probability)
        print("正确类别的负对数概率", -np.log(0.8))
    '''),
    example("梯度下降", "最小化 L(w)=(w-3)²，梯度为 2(w-3)。每次沿梯度反方向更新。学习率过大会振荡甚至发散；梯度不是新的参数，而是更新方向的依据。", '''
        weight = 0.
        learning_rate = 0.2
        for step in range(8):
            gradient = 2 * (weight - 3)
            weight -= learning_rate * gradient
            print(step, round(weight, 4), round((weight - 3) ** 2, 6))
    '''),
], [
    practice("稳定 Softmax", "实现 stable_softmax(x)，沿最后一轴归一化。支持一维和二维 NumPy 浮点数组；必须能处理 [1000,1001] 而不溢出。", 'def stable_softmax(x):\n    raise NotImplementedError', '''
        def stable_softmax(x):
            shifted = x - np.max(x, axis=-1, keepdims=True)
            values = np.exp(shifted)
            return values / values.sum(axis=-1, keepdims=True)
    ''', '''
        np.testing.assert_allclose(stable_softmax(np.array([1000., 1001.])), [0.2689414214, 0.7310585786])
        r = stable_softmax(np.array([[1., 1.], [0., 2.]]))
        assert r.shape == (2, 2)
        np.testing.assert_allclose(r.sum(axis=-1), [1., 1.])
    '''),
    practice("离散期望", "返回数值数组 values 与概率数组 probabilities 的加权和；二者长度相同且概率有效。不要直接取 values.mean()。", 'def expected_value(values, probabilities):\n    raise NotImplementedError', '''
        def expected_value(values, probabilities):
            return np.sum(values * probabilities)
    ''', '''
        assert np.isclose(expected_value(np.array([0., 10.]), np.array([0.8, 0.2])), 2.)
        assert np.isclose(expected_value(np.array([-2., 3., 8.]), np.array([0.2, 0.5, 0.3])), 3.5)
    '''),
    practice("一次梯度更新", "对 (weight-target)² 做一步梯度下降，返回新 weight。输入是浮点数。", 'def gradient_step(weight, target, learning_rate):\n    raise NotImplementedError', '''
        def gradient_step(weight, target, learning_rate):
            return weight - learning_rate * 2 * (weight - target)
    ''', '''
        assert np.isclose(gradient_step(0., 3., 0.1), 0.6)
        assert np.isclose(gradient_step(5., 2., 0.), 5.)
        assert np.isclose(gradient_step(3., 3., 0.1), 3.)
    '''),
], [
    quiz("(4,3) @ (3,7) 的输出形状是什么？", ["(4,7)", "(3,3)", "(4,3,7)"], 0),
    quiz("Softmax 为什么减去每行最大值？", ["改变类别顺序", "避免指数溢出且保持概率不变", "使所有概率为零"], 1),
    quiz("梯度下降更新参数的方向是什么？", ["梯度方向", "随机增大参数", "负梯度方向"], 2),
])
