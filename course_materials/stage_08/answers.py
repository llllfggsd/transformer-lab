"""先独立完成 exercises.py，再阅读参考答案。"""
import math
import numpy as np
import torch
from torch import nn
torch.set_num_threads(1)


def temperature_probs(logits, temperature):
    if temperature <= 0:
        raise ValueError("温度必须为正")
    return torch.softmax(logits / temperature, dim=-1)

def top_k_logits(logits, k):
    threshold = torch.topk(logits, k).values[-1]
    return logits.masked_fill(logits < threshold, float("-inf"))

def freeze_backbone(backbone, head):
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    for parameter in head.parameters():
        parameter.requires_grad_(True)

def run_checks(namespace):
    passed = 0
    checks = ['scores = torch.tensor([[1., 2., 3.]])\ntorch.testing.assert_close(temperature_probs(scores, 2.), torch.softmax(scores / 2, -1))\ntry:\n    temperature_probs(scores, 0.)\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError("温度为零应报错")', 'data = torch.tensor([1., 4., 2., 3.])\nresult = top_k_logits(data, 2)\nassert torch.isneginf(result[[0, 2]]).all()\ntorch.testing.assert_close(result[[1, 3]], torch.tensor([4., 3.]))\ntorch.testing.assert_close(data, torch.tensor([1., 4., 2., 3.]))\ntorch.testing.assert_close(top_k_logits(data, 4), data)', 'base = nn.Linear(4, 4)\nhead = nn.Linear(4, 2)\nfor p in head.parameters():\n    p.requires_grad_(False)\nfreeze_backbone(base, head)\nassert all(not p.requires_grad for p in base.parameters())\nassert all(p.requires_grad for p in head.parameters())']
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
