"""先独立完成 exercises.py，再阅读参考答案。"""
import math
import numpy as np
import torch
from torch import nn
torch.set_num_threads(1)


def make_causal_mask(length):
    return torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)

def scaled_attention(q, k, v, blocked=None):
    scores = q @ k.T / math.sqrt(q.shape[-1])
    if blocked is not None:
        scores = scores.masked_fill(blocked, float("-inf"))
    return scores.softmax(dim=-1) @ v

def to_heads(x, heads):
    b, t, d = x.shape
    return x.reshape(b, t, heads, d // heads).transpose(1, 2)

def run_checks(namespace):
    passed = 0
    checks = ['torch.testing.assert_close(make_causal_mask(3), torch.tensor([[False, True, True], [False, False, True], [False, False, False]]))\nassert make_causal_mask(1).shape == (1, 1) and not make_causal_mask(1).item()', 'q = torch.tensor([[1., 0.]])\nk = torch.eye(2)\nv = torch.tensor([[10., 0.], [0., 20.]])\nexpected = torch.softmax(torch.tensor([[1. / math.sqrt(2), 0.]]), -1) @ v\ntorch.testing.assert_close(scaled_attention(q, k, v), expected)\ntorch.testing.assert_close(scaled_attention(q, k, v, torch.tensor([[False, True]])), v[:1])', 'data = torch.arange(48.).reshape(2, 3, 8)\nresult = to_heads(data, 2)\nassert result.shape == (2, 2, 3, 4)\ntorch.testing.assert_close(result[1, 1, 2], data[1, 2, 4:])']
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
