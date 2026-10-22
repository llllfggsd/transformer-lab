import math
import numpy as np
import torch
from torch import nn
torch.set_num_threads(1)


# 构造训练对
# 给定一维数据 sequence、有效起点列表 starts 和 context，返回堆叠的 (x,y)，形状都是 (len(starts),context)，标签相对输入错开一位。
def training_pairs(sequence, starts, context):
    raise NotImplementedError

# 逐 Token 交叉熵
# logits 是 (B,T,V)，targets 是 (B,T)。返回平均交叉熵标量 Tensor，标签为 ignore_id 的位置不参与。至少有一个有效标签。
def language_model_loss(logits, targets, ignore_id=-100):
    raise NotImplementedError

# 最后位置的贪心选择
# 输入 logits (B,T,V)，返回每个样本最后位置分数最高的 ID，形状必须为 (B,1)。
def greedy_next_token(logits):
    raise NotImplementedError

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
