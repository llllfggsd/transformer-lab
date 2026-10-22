import math
import numpy as np
import torch
from torch import nn
torch.set_num_threads(1)


# 编码文本
# 按字符编码 text，使用传入字典 vocabulary。未知字符使用 unknown_id；返回 Python 列表，空文本返回空列表。
def encode_text(text, vocabulary, unknown_id=1):
    raise NotImplementedError

# 补齐批次
# 输入非空列表 sequences，每项是非空一维整数 Tensor。返回右侧补齐的 (B,T) 张量，使用传入 pad_id，不修改原序列。
def pad_batch(sequences, pad_id=0):
    raise NotImplementedError

# 输入目标对齐
# sequence 是长度至少为 2 的一维 Tensor。返回 (x,y)，x 为除最后一个元素外的部分，y 为除第一个元素外的部分。
def next_token_pair(sequence):
    raise NotImplementedError

def run_checks(namespace):
    passed = 0
    checks = ['assert encode_text("我爱!", {"我": 2, "爱": 3}) == [2, 3, 1]\nassert encode_text("", {}) == []\nassert encode_text("x", {}, 9) == [9]', 'a, b = torch.tensor([2, 3, 4]), torch.tensor([5])\ntorch.testing.assert_close(pad_batch([a, b], 9), torch.tensor([[2, 3, 4], [5, 9, 9]]))\ntorch.testing.assert_close(b, torch.tensor([5]))', 'x, y = next_token_pair(torch.tensor([2, 4, 6, 8]))\ntorch.testing.assert_close(x, torch.tensor([2, 4, 6]))\ntorch.testing.assert_close(y, torch.tensor([4, 6, 8]))']
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
