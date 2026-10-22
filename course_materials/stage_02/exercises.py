import math
import numpy as np


# 稳定 Softmax
# 实现 stable_softmax(x)，沿最后一轴归一化。支持一维和二维 NumPy 浮点数组；必须能处理 [1000,1001] 而不溢出。
def stable_softmax(x):
    raise NotImplementedError

# 离散期望
# 返回数值数组 values 与概率数组 probabilities 的加权和；二者长度相同且概率有效。不要直接取 values.mean()。
def expected_value(values, probabilities):
    raise NotImplementedError

# 一次梯度更新
# 对 (weight-target)² 做一步梯度下降，返回新 weight。输入是浮点数。
def gradient_step(weight, target, learning_rate):
    raise NotImplementedError

def run_checks(namespace):
    passed = 0
    checks = ['np.testing.assert_allclose(stable_softmax(np.array([1000., 1001.])), [0.2689414214, 0.7310585786])\nr = stable_softmax(np.array([[1., 1.], [0., 2.]]))\nassert r.shape == (2, 2)\nnp.testing.assert_allclose(r.sum(axis=-1), [1., 1.])', 'assert np.isclose(expected_value(np.array([0., 10.]), np.array([0.8, 0.2])), 2.)\nassert np.isclose(expected_value(np.array([-2., 3., 8.]), np.array([0.2, 0.5, 0.3])), 3.5)', 'assert np.isclose(gradient_step(0., 3., 0.1), 0.6)\nassert np.isclose(gradient_step(5., 2., 0.), 5.)\nassert np.isclose(gradient_step(3., 3., 0.1), 3.)']
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
