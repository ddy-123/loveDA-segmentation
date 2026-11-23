"""LoveDA 常量与标签处理。

实测（D1 核验）：原始 mask 取值 0..7，其中
    0        no-data，必须忽略
    1..7     背景 / 建筑 / 道路 / 水体 / 裸地 / 林地 / 农田
所以训练时统一做 raw-1 映射到 0..6，把 0 映射到 255(ignore_index)。
"""
from __future__ import annotations

IGNORE_INDEX = 255
NUM_CLASSES = 7

CLASS_NAMES = ["background", "building", "road", "water", "barren", "forest", "agriculture"]
CLASS_NAMES_CN = ["背景", "建筑", "道路", "水体", "裸地", "林地", "农田"]

# mmseg 官方 LoveDA 调色板
PALETTE = [
    (255, 255, 255),
    (255, 0, 0),
    (255, 255, 0),
    (0, 0, 255),
    (159, 129, 183),
    (0, 255, 0),
    (0, 255, 255),
]

# 原始值 -> 训练用类别下标
RAW_TO_IDX = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}

SPLITS = ("Train", "Val")
DOMAINS = ("Urban", "Rural")


def remap_mask(arr):
    """原始 0..7 -> 0..6 / 255。输入输出都是 uint8 numpy 数组。"""
    import numpy as np

    out = np.full(arr.shape, IGNORE_INDEX, dtype=np.uint8)
    for raw, idx in RAW_TO_IDX.items():
        out[arr == raw] = idx
    return out
