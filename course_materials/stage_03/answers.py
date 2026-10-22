"""先独立完成 exercises.py，再阅读参考答案。"""
import math
import numpy as np


def mse_loss(prediction, target):
    return np.mean((prediction - target) ** 2)

def classification_accuracy(logits, labels):
    return np.mean(np.argmax(logits, axis=-1) == labels)

def standardize(train, values):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < 1e-8, 1., std)
    return (values - mean) / std

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
