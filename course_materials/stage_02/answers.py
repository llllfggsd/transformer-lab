"""先独立完成 exercises.py，再阅读参考答案。"""
import math
import numpy as np


def stable_softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=-1, keepdims=True)

def expected_value(values, probabilities):
    return np.sum(values * probabilities)

def gradient_step(weight, target, learning_rate):
    return weight - learning_rate * 2 * (weight - target)

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
