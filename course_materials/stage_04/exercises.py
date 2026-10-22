import math
import numpy as np
import torch
from torch import nn
torch.set_num_threads(1)


# 自动求导的平方梯度
# 实现 square_gradient(values)：输入一维浮点张量，返回 sum(values²) 对各元素的梯度，不修改输入的 requires_grad 或已有 grad。
def square_gradient(values):
    raise NotImplementedError

# 可训练的线性层
# 实现继承 nn.Module 的 LinearRegressor(input_dim)。注册一个 nn.Linear(input_dim,1)，forward 返回 (B,1)。参数必须可被优化器获取。
class LinearRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        raise NotImplementedError
    def forward(self, x):
        raise NotImplementedError

# 训练一步
# 实现 train_step(model, optimizer, x, y)：使用 MSE，清梯度、反向并更新参数，返回本次更新前的损失浮点数。
def train_step(model, optimizer, x, y):
    raise NotImplementedError

def run_checks(namespace):
    passed = 0
    checks = ['values = torch.tensor([-2., 0., 3.])\ntorch.testing.assert_close(square_gradient(values), torch.tensor([-4., 0., 6.]))\nassert not values.requires_grad and values.grad is None', 'model_under_test = LinearRegressor(3)\nassert isinstance(model_under_test, nn.Module)\noutput = model_under_test(torch.ones(4, 3))\nassert output.shape == (4, 1)\nassert sum(p.numel() for p in model_under_test.parameters()) == 4\noutput.sum().backward()\nassert all(p.grad is not None for p in model_under_test.parameters())', 'layer = nn.Linear(1, 1, bias=False)\nwith torch.no_grad():\n    layer.weight.fill_(0.)\nopt = torch.optim.SGD(layer.parameters(), lr=0.1)\nvalue = train_step(layer, opt, torch.ones(2, 1), torch.ones(2, 1))\nassert abs(value - 1.) < 1e-6\ntorch.testing.assert_close(layer.weight, torch.tensor([[0.2]]))\ntrain_step(layer, opt, torch.ones(2, 1), torch.ones(2, 1))\ntorch.testing.assert_close(layer.weight, torch.tensor([[0.36]]))']
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
