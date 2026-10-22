import math
import numpy as np
import torch
from torch import nn
torch.set_num_threads(1)


# RMSNorm 核心
# 沿最后一维实现无仿射参数 RMSNorm：x / sqrt(mean(x²)+eps)。保留形状，默认 eps=1e-6。
def normalize_rms(x, eps=1e-6):
    raise NotImplementedError

# 局部因果掩码
# 返回 (length,length) 布尔掩码。Query i 只看 max(0,i-window+1) 到 i，其余为 True。length 与 window 为正整数。
def local_causal_mask(length, window):
    raise NotImplementedError

# 对称量化
# 返回 (quantized,scale)，quantized 为 int8，scale 为正的标量 Tensor。scale=max(max(abs(x)),1e-8)/127，先四舍五入再裁到 [-127,127]。
def quantize_int8(x):
    raise NotImplementedError

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
