"""先独立完成 exercises.py，再阅读参考答案。"""
import math
import numpy as np
import torch
from torch import nn
torch.set_num_threads(1)


def training_pairs(sequence, starts, context):
    x = torch.stack([sequence[i:i + context] for i in starts])
    y = torch.stack([sequence[i + 1:i + context + 1] for i in starts])
    return x, y

def language_model_loss(logits, targets, ignore_id=-100):
    return nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=ignore_id)

def greedy_next_token(logits):
    return logits[:, -1].argmax(dim=-1, keepdim=True)

def run_checks(namespace):
    passed = 0
    checks = ['x, y = training_pairs(torch.arange(8), [0, 3], 3)\ntorch.testing.assert_close(x, torch.tensor([[0, 1, 2], [3, 4, 5]]))\ntorch.testing.assert_close(y, torch.tensor([[1, 2, 3], [4, 5, 6]]))', 'logits = torch.tensor([[[2., 0.], [0., 2.]]], requires_grad=True)\ntargets = torch.tensor([[0, -100]])\nloss = language_model_loss(logits, targets)\ntorch.testing.assert_close(loss, nn.functional.cross_entropy(logits[:, 0], torch.tensor([0])))\nloss.backward()\nassert logits.grad[:, 1].abs().sum() == 0', 'scores = torch.tensor([[[9., 0.], [0., 3.]], [[0., 5.], [4., 1.]]])\ntorch.testing.assert_close(greedy_next_token(scores), torch.tensor([[1], [0]]))']
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
