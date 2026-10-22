import math
import numpy as np
import torch
from torch import nn
torch.set_num_threads(1)


# 温度概率
# temperature_probs(logits, temperature) 沿最后一轴返回 Softmax(logits/temperature)。温度 <=0 时抛出 ValueError。
def temperature_probs(logits, temperature):
    raise NotImplementedError

# Top-k 过滤
# logits 是一维且分数无并列，1<=k<=词表大小。返回同形状张量，仅保留最高 k 项原值，其余为 -inf，不修改输入。
def top_k_logits(logits, k):
    raise NotImplementedError

# 只训练指定的分类头
# freeze_backbone(backbone, head) 让 backbone 的全部参数停止求导，让 head 的参数可求导；原地修改，不必返回值。
def freeze_backbone(backbone, head):
    raise NotImplementedError

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
