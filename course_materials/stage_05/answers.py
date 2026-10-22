"""先独立完成 exercises.py，再阅读参考答案。"""
import math
import numpy as np
import torch
from torch import nn
torch.set_num_threads(1)


def encode_text(text, vocabulary, unknown_id=1):
    return [vocabulary.get(token, unknown_id) for token in text]

def pad_batch(sequences, pad_id=0):
    return nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=pad_id)

def next_token_pair(sequence):
    return sequence[:-1], sequence[1:]

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
