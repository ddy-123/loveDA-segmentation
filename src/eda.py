"""EDA：样例图墙（图 + mask 叠加）、类别占比、共现矩阵、跨域色调对比。

图内文字一律用英文，避免 matplotlib 中文字体缺失出豆腐块；报告正文再写中文。
输出到 outputs/figures/。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from src.loveda import CLASS_NAMES, PALETTE, RAW_TO_IDX  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "LoveDA"
FIG = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"


def colorize(mask_raw: np.ndarray) -> np.ndarray:
    """原始 mask(0..7) -> RGB 彩色图，no-data 画黑。"""
    rgb = np.zeros((*mask_raw.shape, 3), dtype=np.uint8)
    for raw, idx in RAW_TO_IDX.items():
        rgb[mask_raw == raw] = PALETTE[idx]
    return rgb


def overlay(img: np.ndarray, mask_raw: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    col = colorize(mask_raw)
    out = img.copy()
    valid = mask_raw != 0
    out[valid] = (img[valid] * (1 - alpha) + col[valid] * alpha).astype(np.uint8)
    return out


def gallery(n_per_domain: int, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    cells = []
    for domain in ("Urban", "Rural"):
        imgs = sorted((RAW / "Train" / domain / "images_png").glob("*.png"))
        if not imgs:
            continue
        msk_all = RAW / "Train" / domain / "masks_png"
        imgs = [p for p in imgs if (msk_all / p.name).exists()]
        if not imgs:
            continue
        pick = rng.choice(len(imgs), size=min(n_per_domain, len(imgs)), replace=False)
        for i in pick:
            p = imgs[int(i)]
            m = msk_all / p.name
            with Image.open(p) as im:
                img = np.asarray(im.convert("RGB"))
            with Image.open(m) as mk:
                raw = np.asarray(mk.convert("L"))
            cells.append((domain, p.name, img, raw))

    n = len(cells)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (domain, name, img, raw) in zip(axes, cells):
        ax.imshow(overlay(img, raw))
        nodata = float((raw == 0).mean())
        ax.set_title(f"{domain} | {name}\nnodata={nodata:.1%}", fontsize=8, pad=3)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=np.array(c) / 255) for c in PALETTE]
    fig.legend(handles, CLASS_NAMES, loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig.suptitle("LoveDA samples: image + mask overlay (Train)", fontsize=12)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.05, hspace=0.22, wspace=0.05)
    out = FIG / "eda_gallery.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def class_ratio_chart() -> Path | None:
    f = TABLES / "class_pixel_ratio.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    cols = [f"raw{i}_ratio" for i in range(8)]
    labels = ["nodata"] + CLASS_NAMES
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(labels))
    w = 0.8 / len(df)
    for k, (_, row) in enumerate(df.iterrows()):
        ax.bar(x + k * w - 0.4, [row[c] for c in cols], width=w, label=f"{row['split']}/{row['domain']}")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("pixel ratio")
    ax.set_title("LoveDA raw class pixel ratio by split/domain")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG / "eda_class_ratio.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def cooccurrence(split: str, domain: str, max_images: int = 400) -> Path | None:
    img_dir = RAW / split / domain / "images_png"
    msk_dir = RAW / split / domain / "masks_png"
    if not img_dir.is_dir():
        return None
    files = [p for p in sorted(img_dir.glob("*.png")) if (msk_dir / p.name).exists()][:max_images]
    if not files:
        return None
    mat = np.zeros((7, 7), dtype=np.int64)
    for p in files:
        with Image.open(msk_dir / p.name) as mk:
            raw = np.asarray(mk.convert("L"))
        present = [RAW_TO_IDX[v] for v in np.unique(raw) if v in RAW_TO_IDX]
        for a in present:
            for b in present:
                mat[a, b] += 1
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="viridis")
    ax.set_xticks(range(7), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(7), CLASS_NAMES)
    ax.set_title(f"Class co-occurrence per image ({split}/{domain}, n={len(files)})")
    for i in range(7):
        for j in range(7):
            ax.text(j, i, mat[i, j], ha="center", va="center", fontsize=7, color="w")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    out = FIG / f"eda_cooccur_{split}_{domain}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def domain_stats_chart() -> Path | None:
    f = TABLES / "image_level_stats.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for lab, sub in df.groupby(df["split"] + "/" + df["domain"]):
        axes[0].hist(sub["brightness"], bins=50, alpha=0.5, label=lab, density=True)
        axes[1].hist(sub["hue_mean"], bins=50, alpha=0.5, label=lab, density=True)
    axes[0].set_title("Brightness distribution")
    axes[0].set_xlabel("mean pixel value")
    axes[1].set_title("Hue distribution")
    axes[1].set_xlabel("mean hue")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    out = FIG / "eda_domain_tone.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=8, help="每个 domain 抽几张")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)

    made = []
    if (RAW / "Train" / "Urban" / "images_png").is_dir():
        made.append(gallery(args.samples, args.seed))
    made.append(class_ratio_chart())
    made.append(cooccurrence("Train", "Urban"))
    made.append(cooccurrence("Train", "Rural"))
    made.append(domain_stats_chart())
    for p in made:
        if p:
            print("figure ->", p)
    (FIG / "eda_index.json").write_text(
        json.dumps([str(p.name) for p in made if p], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
