import math
import numpy as np


# 均方误差
# 返回同形状 NumPy 数组 prediction 和 target 的均方误差标量。
def mse_loss(prediction, target):
    raise NotImplementedError

# 分类准确率
# logits 形状 (B,C)，labels 形状 (B,)。取最后一轴 argmax 作为预测，返回准确率。
def classification_accuracy(logits, labels):
    raise NotImplementedError

# 训练统计量标准化
# 用 train 的逐列均值和标准差标准化 values；标准差小于 1e-8 时使用 1，避免除零。返回与 values 同形状的数组。
def standardize(train, values):
    raise NotImplementedError

def run_checks(namespace):
    passed = 0
    checks = ['assert np.isclose(mse_loss(np.array([1., 3.]), np.array([2., 1.])), 2.5)\nassert mse_loss(np.ones((2, 3)), np.ones((2, 3))) == 0', 'scores = np.array([[1., 3., 0.], [4., 1., 2.], [0., 1., 5.]])\nassert np.isclose(classification_accuracy(scores, np.array([1, 0, 0])), 2 / 3)\nassert classification_accuracy(np.array([[0., 1.]]), np.array([1])) == 1', 'np.testing.assert_allclose(standardize(np.array([[1., 2.], [3., 2.]]), np.array([[5., 2.]])), [[3., 0.]])\nassert standardize(np.ones((3, 2)), np.ones((2, 2))).shape == (2, 2)']
    for index, code in enumerate(checks, 1):
        try:
            exec(code, dict(namespace))
        except Exception as error:
            print(f'练习 {index}：需要修改（{type(error).__name__}: {error}）')
        else:
            passed += 1
            print(f'练习 {index}：通过')
    print(f'完成 {passed}/{len(checks)} 题')
    return passed == len(checks)


if __name__ == '__main__':
    raise SystemExit(0 if run_checks(globals()) else 1)
