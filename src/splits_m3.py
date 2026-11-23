"""M3 实验的划分：同模型、同超参、同种子、同训练样本量，唯一变量是"分布"。

为什么要对齐样本量：
  随机划分的池子是 Train+Val 合并（3352），官方划分只有 Train（2522）。
  若直接用各自的池子，"随机划分赢"就分不清是同分布还是多用了数据。
  所以两臂都取 2271 张训练 + 251 张 dev，epoch 数也一致（12000 iter × 等效 batch 8）。

  A 随机划分  训练池 = random_train.txt（Train+Val 合并后按域分层 8:2 的 80%）
              从池中抽 2271 张训练，再抽 251 张 dev
              测试 = random_test.txt（839 张，训练时没见过）
  B 官方划分  训练池 = train.txt（官方 Train，跨城）
              2271 训练 + 251 dev
              测试 = val.txt（官方 Val 1669 张，没见过）

用法：python -m src.splits_m3
"""
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
SEED = 3407
N_TRAIN = 2271      # = 官方 Train 2522 - dev 251，两臂都用这个数
N_DEV = 251


def read(name: str) -> list[str]:
    return [ln.strip() for ln in (SPLITS / name).read_text(encoding="utf-8").splitlines() if ln.strip()]


def write(name: str, items: list[str]) -> None:
    items = sorted(items)
    (SPLITS / name).write_text("\n".join(items) + "\n", encoding="utf-8")
    print(f"{name:18s} {len(items):5d} 张")


def domain_of(p: str) -> str:
    return "Urban" if "/Urban/" in p else "Rural"


def main() -> int:
    rng = random.Random(SEED)

    # B 官方：全部 Train，切 251 作 dev
    pool_b = sorted(read("train.txt"))
    rng.shuffle(pool_b)
    b_dev, b_train = sorted(pool_b[:N_DEV]), sorted(pool_b[N_DEV:])
    assert len(b_train) == N_TRAIN, len(b_train)
    print("B 官方划分")
    write("m3o_train.txt", b_train)
    write("m3o_dev.txt", b_dev)

    # A 随机：从合并池抽同样多（按域分层，保持 Urban/Rural 比例与 B 一致）
    pool_a = sorted(read("random_train.txt"))
    ratio_b = Counter(domain_of(p) for p in b_train)
    a_train: list[str] = []
    for domain, n in sorted(ratio_b.items()):
        cand = [p for p in pool_a if domain_of(p) == domain]
        rng.shuffle(cand)
        a_train += cand[:n]
        print(f"  A {domain:5s} 池 {len(cand):4d} -> 取 {n}")
    a_train = sorted(a_train)
    rest = [p for p in pool_a if p not in set(a_train)]
    rng.shuffle(rest)
    a_dev = sorted(rest[:N_DEV])
    print("A 随机划分")
    write("m3r_train.txt", a_train)
    write("m3r_dev.txt", a_dev)

    test_a, test_b = read("random_test.txt"), read("val.txt")
    print(f"测试集: A=random_test {len(test_a)} / B=val {len(test_b)}")

    leak_a = set(a_train) & set(test_a)
    leak_b = set(b_train) & set(test_b)
    leak_ab = set(a_train) & set(b_train)
    print(f"泄漏检查: A∩测试 {len(leak_a)} / B∩测试 {len(leak_b)} / A∩B训练 {len(leak_ab)}")
    assert not leak_a and not leak_b, "训练集与测试集重叠"

    print(f"\n两臂对齐检查: train {len(a_train)} vs {len(b_train)} / dev {len(a_dev)} vs {len(b_dev)}")
    for tag, items in (("A 随机", a_train), ("B 官方", b_train)):
        c = Counter(domain_of(p) for p in items)
        print(f"  {tag} 训练集域构成: Urban {c['Urban']} / Rural {c['Rural']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
