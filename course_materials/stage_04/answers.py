"""先独立完成 exercises.py，再阅读参考答案。"""
import math
import numpy as np
import torch
from torch import nn
torch.set_num_threads(1)


def square_gradient(values):
    leaf = values.detach().clone().requires_grad_(True)
    (leaf ** 2).sum().backward()
    return leaf.grad

class LinearRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.linear(x)

def train_step(model, optimizer, x, y):
    optimizer.zero_grad()
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()
    return loss.item()

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
