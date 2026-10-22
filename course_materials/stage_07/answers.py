"""先独立完成 exercises.py，再阅读参考答案。"""
import math
import numpy as np
import torch
from torch import nn
torch.set_num_threads(1)


def position_encoding(length, dimension):
    pos = torch.arange(length).float().unsqueeze(1)
    freq = torch.exp(torch.arange(0, dimension, 2).float() * (-math.log(10000.) / dimension))
    result = torch.zeros(length, dimension)
    result[:, 0::2] = torch.sin(pos * freq)
    result[:, 1::2] = torch.cos(pos * freq)
    return result

def residual_layer_norm(x, branch):
    return nn.functional.layer_norm(x + branch, (x.shape[-1],), eps=1e-5)

def build_ffn(dimension, hidden):
    return nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(), nn.Linear(hidden, dimension))

def run_checks(namespace):
    passed = 0
    checks = ['result = position_encoding(3, 4)\nassert result.shape == (3, 4)\ntorch.testing.assert_close(result[0], torch.tensor([0., 1., 0., 1.]))\ntorch.testing.assert_close(result[1], torch.tensor([math.sin(1), math.cos(1), math.sin(0.01), math.cos(0.01)]))', 'x = torch.tensor([[[1., 2., 3., 4.]]])\nb = torch.tensor([[[2., 0., 1., 0.]]])\ntorch.testing.assert_close(residual_layer_norm(x, b), nn.functional.layer_norm(x + b, (4,)))\ntorch.testing.assert_close(residual_layer_norm(torch.ones(2, 3), torch.ones(2, 3)), torch.zeros(2, 3))', 'f = build_ffn(6, 18)\nassert isinstance(f, nn.Sequential) and isinstance(f[1], nn.GELU)\nassert f(torch.randn(2, 4, 6)).shape == (2, 4, 6)\nassert sum(p.numel() for p in f.parameters()) == 240']
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
