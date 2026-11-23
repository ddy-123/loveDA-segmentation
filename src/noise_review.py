"""标注噪声复核表：挑出模型错得最多的区域，裁成小块给人看，附一个待填的 CSV。

产物：
  outputs/figures/noise_review.png   25 个错例小块（4×? 网格，标注 GT / 预测）
  outputs/tables/noise_review.csv    待人工判定的表格（verdict 列留空）

判定口径（人工填）：model = 模型错；label = 标签错；ambiguous = 边界/尺度本身就模糊

用法：
  python -m src.noise_review --model segformer_b2 --ckpt outputs/ckpt/m3_official/best.pth \
      --split val --limit 80 --n 25
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi

from src.dataset import LoveDADataset, read_split
from src.evaluate import sliding_infer
from src.loveda import CLASS_NAMES
from src.models import build_model

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"
CROP = 160


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--tag", default="noise")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = state.get("cfg") or {}
    model = build_model(args.model, pretrained=False, arch=cfg.get("pretrained")).to(device)
    model.load_state_dict(state.get("model", state), strict=True)
    model.eval()

    items = read_split(ROOT / "data" / "splits" / f"{args.split}.txt")[: args.limit]
    ds = LoveDADataset(items, crop=None, train=False)

    # 收集"错误连通域"：同一错误类别的连通块，取最大的一些
    cands = []
    for i in range(len(ds)):
        s = ds[i]
        x = s["image"].unsqueeze(0).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            out = sliding_infer(model, x, 512, 256)
        pred = out.argmax(1)[0].cpu().numpy().astype(np.int64)
        gt = s["mask"].numpy().astype(np.int64)
        valid = gt != 255
        err = (pred != gt) & valid
        if err.sum() == 0:
            continue
        lab, num = ndi.label(err)
        if num == 0:
            continue
        sizes = ndi.sum(err, lab, range(1, num + 1))
        for k in np.argsort(-sizes)[:2]:           # 每张图取最大两个错误块
            comp = lab == (k + 1)
            if comp.sum() < 400:
                continue
            ys, xs = np.where(comp)
            cy, cx = int(ys.mean()), int(xs.mean())
            y0 = max(0, min(1024 - CROP, cy - CROP // 2))
            x0 = max(0, min(1024 - CROP, cx - CROP // 2))
            # 该块里最常见的 GT / 预测（去重掉正确像素）
            g = np.bincount(gt[comp].ravel(), minlength=8)[:8].copy()
            g[0] = 0
            p = np.bincount(pred[comp].ravel(), minlength=8)[:8].copy()
            p[0] = 0
            raw = (s["image"].numpy().transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            raw = np.clip(raw * 255, 0, 255).astype(np.uint8)
            cands.append(
                {
                    "image": s["name"],
                    "y": y0,
                    "x": x0,
                    "size": int(comp.sum()),
                    "gt_class": CLASS_NAMES[int(g.argmax()) - 1],
                    "pred_class": CLASS_NAMES[int(p.argmax()) - 1],
                    "img": raw[y0 : y0 + CROP, x0 : x0 + CROP].copy(),
                    "gt": gt[y0 : y0 + CROP, x0 : x0 + CROP].copy(),
                    "pred": pred[y0 : y0 + CROP, x0 : x0 + CROP].copy(),
                    "verdict": "",
                }
            )

    cands.sort(key=lambda c: -c["size"])
    sel = cands[: args.n]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(sel)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, sel):
        ax.imshow(c["img"])
        ax.set_title(f"GT={c['gt_class']}\nP={c['pred_class']}  {c['size']}px", fontsize=7)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("label-noise review: worst error regions (fill noise_review.csv)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_fig = FIG / f"noise_review_{args.tag}.png"
    fig.savefig(out_fig, dpi=130)
    plt.close(fig)

    out_csv = TABLES / f"noise_review_{args.tag}.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["id", "image", "y", "x", "size", "gt_class", "pred_class", "verdict"])
        w.writeheader()
        for i, c in enumerate(sel, 1):
            w.writerow({k: c[k] for k in ["image", "y", "x", "size", "gt_class", "pred_class", "verdict"]} | {"id": i})

    print(f"候选错误块 {len(cands)} 个，取前 {n} 个")
    print("图 ->", out_fig)
    print("表 ->", out_csv, "（verdict 列填 model / label / ambiguous）")
    from collections import Counter

    print("这些块的 GT/预测类对:", Counter((c["gt_class"], c["pred_class"]) for c in sel).most_common(5))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
