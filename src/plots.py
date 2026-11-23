"""报告/PPT 用的结果图：从 outputs/metrics/*.json 生成，保证图和表同源。

用法：python -m src.plots --models unet deeplabv3plus --split dev
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.loveda import CLASS_NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs" / "metrics"
FIG = ROOT / "outputs" / "figures"

LABELS = {
    "unet": "B1 UNet (scratch)",
    "deeplabv3plus": "B2 DeepLabV3+ (IN)",
    "segformer_b2": "M1 SegFormer-B2 (ADE)",
    "m2_color": "M2 + color aug",
}

# 带后缀的实验（文件名 eval_<model>_<split>_<tag>.json）在这里登记
TAGGED = {
    "m2_color": ("segformer_b2", "m2"),
    "m1_last": ("segformer_b2", "last"),
}


def eval_path(key: str, split: str) -> Path:
    if key in TAGGED:
        model, tag = TAGGED[key]
        return METRICS / f"eval_{model}_{split}_{tag}.json"
    return METRICS / f"eval_{key}_{split}.json"


def load(name: str, split: str) -> dict:
    return json.loads(eval_path(name, split).read_text(encoding="utf-8"))


def plot_miou(models: list[str], split: str) -> Path:
    data = [load(m, split) for m in models]
    names = [LABELS.get(m, m) for m in models]
    vals = [d["miou"] for d in data]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = ["#8c8c8c", "#2f6db5", "#2f9e6b", "#d9822b", "#7b4fa8", "#c0392b"]
    bars = ax.bar(names, vals, color=colors[: len(models)])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.4f}", ha="center", fontsize=10)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_ylabel("mIoU (512/256 sliding window)")
    ax.set_title(f"LoveDA {split}: mIoU comparison")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG / f"results_miou_{split}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_per_class(models: list[str], split: str) -> Path:
    data = [load(m, split) for m in models]
    x = np.arange(len(CLASS_NAMES))
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, (m, d) in enumerate(zip(models, data)):
        ax.bar(x + i * w - 0.4 + w / 2, d["per_class_iou"], width=w, label=LABELS.get(m, m))
    ax.set_xticks(x, CLASS_NAMES, rotation=25, ha="right")
    ax.set_ylabel("IoU")
    ax.set_title(f"Per-class IoU ({split})")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG / f"results_per_class_{split}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_confusion(model: str, split: str) -> Path:
    d = load(model, split)
    cm = np.array(d["confusion"], dtype=np.float64)
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title(f"Normalized confusion matrix: {LABELS.get(model, model)} ({split})")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if cmn[i, j] > 0.01:
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="w" if cmn[i, j] > 0.5 else "k")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    out = FIG / f"results_confusion_{model}_{split}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["unet", "deeplabv3plus"])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--confusion", action="store_true", help="额外画混淆矩阵")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)

    out = [plot_miou(args.models, args.split), plot_per_class(args.models, args.split)]
    if args.confusion:
        out += [plot_confusion(m, args.split) for m in args.models]
    for p in out:
        print("figure ->", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
