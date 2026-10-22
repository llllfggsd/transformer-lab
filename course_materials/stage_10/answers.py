"""先独立完成 exercises.py，再阅读参考答案。"""
import math
import numpy as np
import torch
from torch import nn
torch.set_num_threads(1)


def normalize_rms(x, eps=1e-6):
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)

def local_causal_mask(length, window):
    i = torch.arange(length).unsqueeze(1)
    j = torch.arange(length).unsqueeze(0)
    return (j > i) | (j < i - window + 1)

def quantize_int8(x):
    scale = x.abs().max().clamp_min(1e-8) / 127
    return (x / scale).round().clamp(-127, 127).to(torch.int8), scale

def run_checks(namespace):
    passed = 0
    checks = ['x = torch.tensor([[1., 2., 3.], [0., 0., 0.]])\ntorch.testing.assert_close(normalize_rms(x), x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6))\nassert torch.isfinite(normalize_rms(torch.zeros(2, 4))).all()', 'torch.testing.assert_close(local_causal_mask(4, 2), torch.tensor([[False, True, True, True], [False, False, True, True], [True, False, False, True], [True, True, False, False]]))\nassert (~local_causal_mask(3, 1)).sum() == 3', 'x = torch.tensor([-1., 0., 1.])\nq, scale = quantize_int8(x)\nassert q.dtype == torch.int8 and scale > 0\ntorch.testing.assert_close(q, torch.tensor([-127, 0, 127], dtype=torch.int8))\ntorch.testing.assert_close(q.float() * scale, x)\nq, scale = quantize_int8(torch.zeros(4))\nassert (q == 0).all() and scale > 0']
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
