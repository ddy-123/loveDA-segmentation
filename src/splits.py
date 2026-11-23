"""生成 split 文件。

背景：LoveDA 的 Test 只公开影像、不公开标注（榜单评测），所以不能直接当测试集。
方案（方案文档 3.3 的预案）：
  - train.txt  官方 Train 全部 2522 张（Urban 1156 + Rural 1366）
  - val.txt    官方 Val 全部 1669 张（仅用于监控训练，不参与最终数字）
  - dev.txt    Val 按 domain 分层抽一半，用于选模型/调参
  - test.txt   Val 另一半，只用于最终出数
  - random_train.txt / random_test.txt   M3 实验：Train+Val 合并后按 domain 分层 8:2 随机划分

用法：python -m src.splits
"""
from __future__ import annotations

import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "LoveDA"
SPLITS = ROOT / "data" / "splits"
SEED = 3407


def list_rel(split: str, domain: str) -> list[str]:
    d = RAW / split / domain / "images_png"
    if not d.is_dir():
        return []
    return sorted(f"{split}/{domain}/images_png/{p.name}" for p in d.glob("*.png"))


def write(name: str, items: list[str]) -> Path:
    SPLITS.mkdir(parents=True, exist_ok=True)
    p = SPLITS / name
    p.write_text("\n".join(items) + "\n", encoding="utf-8")
    print(f"{name:18s} {len(items):5d} 张")
    return p


def main() -> int:
    rng = random.Random(SEED)
    train = list_rel("Train", "Urban") + list_rel("Train", "Rural")
    val = list_rel("Val", "Urban") + list_rel("Val", "Rural")
    if not train or not val:
        raise SystemExit(f"数据未就绪：{RAW}")

    write("train.txt", train)
    write("val.txt", val)

    # Val 按 domain 分层对半切
    dev, test = [], []
    for domain in ("Urban", "Rural"):
        items = list_rel("Val", domain)
        rng.shuffle(items)
        half = len(items) // 2
        dev += items[:half]
        test += items[half:]
    write("dev.txt", sorted(dev))
    write("test.txt", sorted(test))

    # M3：合并后按 domain 分层 8:2
    rnd_train, rnd_test = [], []
    for domain in ("Urban", "Rural"):
        items = list_rel("Train", domain) + list_rel("Val", domain)
        rng.shuffle(items)
        cut = int(len(items) * 0.8)
        rnd_train += items[:cut]
        rnd_test += items[cut:]
    write("random_train.txt", sorted(rnd_train))
    write("random_test.txt", sorted(rnd_test))

    (SPLITS / "README.txt").write_text(
        f"""LoveDA 划分（seed={SEED}）
train.txt       官方 Train 全量，训练用
val.txt         官方 Val 全量，仅监控
dev.txt         官方 Val 按 Urban/Rural 分层的一半，选模型/调参
test.txt        官方 Val 另一半，只在最后出数时用
random_train/test.txt   M3 对照：Train+Val 合并后按 domain 分层 8:2
原因：官方 Test 不公开标注，见方案文档 3.3。
""",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
