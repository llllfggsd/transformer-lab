import math
import numpy as np
import torch
from torch import nn
torch.set_num_threads(1)


# 因果掩码
# 返回 (length,length) 布尔 Tensor，对角线以上为 True，其余 False。length 为正整数。
def make_causal_mask(length):
    raise NotImplementedError

# 缩放点积注意力
# q/k/v 为二维浮点 Tensor；返回注意力输出。可选 blocked 是布尔 (查询数,Key数)，True 表示屏蔽；每行至少一个有效 Key。
def scaled_attention(q, k, v, blocked=None):
    raise NotImplementedError

# 拆分注意力头
# x 形状 (B,T,D)，D 可被 heads 整除。返回 (B,heads,T,D/heads)，保持正确的元素对应关系。
def to_heads(x, heads):
    raise NotImplementedError

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
