"""第一阶段参考答案。建议先自己完成 03_practice.py，再阅读本文件。"""

import json
from pathlib import Path
import runpy

import numpy as np


def score_label(score):
    if score >= 60:
        return "通过"
    return "继续练习"


def passing_scores(scores):
    return [score for score in scores if score >= 60]


def mean_score(scores):
    if len(scores) == 0:
        raise ValueError("分数列表不能为空")
    return sum(scores) / len(scores)


def count_words(text):
    counts = {}
    for word in text.split():
        counts[word] = counts.get(word, 0) + 1
    return counts


class TopicTracker:
    def __init__(self, name):
        self.name = name
        self.topics = []

    def complete(self, topic):
        self.topics.append(topic)
        return len(self.topics)


def save_and_load_json(record, path):
    with path.open("w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def row_means(x):
    return x.mean(axis=1)


def linear_forward(x, weights, bias):
    return x @ weights + bias


if __name__ == "__main__":
    practice = runpy.run_path(str(Path(__file__).with_name("03_practice.py")))
    raise SystemExit(0 if practice["run_checks"](globals()) else 1)
