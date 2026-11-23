"""D5 错误分析：四个角度，全部从"预测 vs 真值"直接算。

  1. 边界带    用真值的类别边界做距离变换，按 0-1 / 1-3 / 3-5 / >5 px 分桶算 IoU
  2. 小目标    按 GT 连通域面积分桶（<100 / 100-500 / 500-2000 / >2000 px），统计"被完整召回的物体比例"
  3. 混淆对    混淆矩阵 Top-5 类别对（真实类 -> 预测类）
  4. 错例三联图 按每张图的错误像素数排序，输出最差的若干张（影像/真值/预测）

用法：
  python -m src.error_analysis --model segformer_b2 --ckpt outputs/ckpt/segformer_b2/best.pth \
      --split dev --limit 200 --tag m1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from tqdm import tqdm

from src.dataset import LoveDADataset, read_split
from src.evaluate import colorize, sliding_infer
from src.loveda import CLASS_NAMES, CLASS_NAMES_CN, IGNORE_INDEX
from src.models import build_model

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
METRICS = ROOT / "outputs" / "metrics"
FIG = ROOT / "outputs" / "figures"


def boundary_mask(gt: np.ndarray) -> np.ndarray:
    """真值里类别发生变化的位置 = 边界（4 邻域）。"""
    b = np.zeros_like(gt, dtype=bool)
    b[:, 1:] |= gt[:, 1:] != gt[:, :-1]
    b[1:, :] |= gt[1:, :] != gt[:-1, :]
    return b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--tag", default="")
    ap.add_argument("--num-bad", type=int, default=6, help="输出几张最差的三联图")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = state.get("cfg") or {}
    model = build_model(args.model, pretrained=False, arch=cfg.get("pretrained")).to(device)
    model.load_state_dict(state.get("model", state), strict=True)
    model.eval()

    items = read_split(ROOT / "data" / "splits" / f"{args.split}.txt")[: args.limit]
    ds = LoveDADataset(items, crop=None, train=False)
    n_cls = len(CLASS_NAMES)

    # 分桶统计
    dist_bins = [(0, 1), (1, 3), (3, 5), (5, 10**9)]
    bin_inter = {b: np.zeros(n_cls) for b in dist_bins}
    bin_union = {b: np.zeros(n_cls) for b in dist_bins}
    area_bins = [(0, 100), (100, 500), (500, 2000), (2000, 10**12)]
    obj_total = {b: 0 for b in area_bins}
    obj_hit = {b: 0 for b in area_bins}
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    per_image_err: list[tuple[float, str, np.ndarray, np.ndarray, np.ndarray]] = []

    for i in tqdm(range(len(ds)), desc="error analysis"):
        s = ds[i]
        x = s["image"].unsqueeze(0).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            out = sliding_infer(model, x, 512, 256)
        pred = out.argmax(1)[0].cpu().numpy().astype(np.int64)
        gt = s["mask"].numpy().astype(np.int64)
        valid = gt != IGNORE_INDEX

        # 混淆矩阵
        idx = gt[valid] * n_cls + pred[valid]
        cm += np.bincount(idx, minlength=n_cls**2).reshape(n_cls, n_cls)

        # 1) 边界距离分桶
        bnd = boundary_mask(gt)
        if bnd.any():
            dist = ndi.distance_transform_edt(~bnd)
        else:
            dist = np.full(gt.shape, 1e9)
        for lo, hi in dist_bins:
            m = valid & (dist >= lo) & (dist < hi)
            if not m.any():
                continue
            for c in range(n_cls):
                t = m & (gt == c)
                if t.any():
                    bin_inter[(lo, hi)][c] += (t & (pred == c)).sum()
                    bin_union[(lo, hi)][c] += t.sum()

        # 2) 小目标：GT 连通域
        for c in range(n_cls):
            g = (gt == c) & valid
            if not g.any():
                continue
            lab, num = ndi.label(g)
            if num == 0:
                continue
            areas = ndi.sum(g, lab, range(1, num + 1))
            hits = ndi.sum((pred == c) & g, lab, range(1, num + 1))
            for a, h in zip(areas, hits):
                for lo, hi in area_bins:
                    if lo <= a < hi:
                        obj_total[(lo, hi)] += 1
                        if a > 0 and h / a >= 0.5:   # 一半以上像素对，算召回
                            obj_hit[(lo, hi)] += 1
                        break

        err = float(((pred != gt) & valid).sum())
        raw = (s["image"].numpy().transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        raw = np.clip(raw * 255, 0, 255).astype(np.uint8)
        per_image_err.append((err, s["name"], raw, gt, pred))

    # ---------------- 汇总
    run = f"{args.model}_{args.split}" + (f"_{args.tag}" if args.tag else "")

    boundary_rows = []
    for lo, hi in dist_bins:
        iou = np.divide(bin_inter[(lo, hi)], bin_union[(lo, hi)],
                        out=np.full(n_cls, np.nan), where=bin_union[(lo, hi)] > 0)
        label = f"{lo}-{hi if hi < 1e8 else 'inf'}px"
        boundary_rows.append(
            {
                "bucket": label,
                "miou": float(np.nanmean(iou)),
                "pixels": int(bin_union[(lo, hi)].sum()),
                **{f"iou_{CLASS_NAMES[c]}": (None if np.isnan(iou[c]) else round(float(iou[c]), 4)) for c in range(n_cls)},
            }
        )

    area_rows = []
    for lo, hi in area_bins:
        tot, hit = obj_total[(lo, hi)], obj_hit[(lo, hi)]
        area_rows.append(
            {
                "area_bucket": f"{lo}-{hi if hi < 1e11 else 'inf'}px",
                "objects": tot,
                "recalled": hit,
                "recall": round(hit / tot, 4) if tot else None,
            }
        )

    pairs = []
    for i in range(n_cls):
        for j in range(n_cls):
            if i != j and cm[i, j] > 0:
                pairs.append({"true": CLASS_NAMES[i], "pred": CLASS_NAMES[j], "pixels": int(cm[i, j])})
    pairs.sort(key=lambda x: -x["pixels"])

    import csv

    with (TABLES / f"err_boundary_{run}.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(boundary_rows[0].keys()))
        w.writeheader()
        w.writerows(boundary_rows)
    with (TABLES / f"err_smallobj_{run}.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(area_rows[0].keys()))
        w.writeheader()
        w.writerows(area_rows)
    with (TABLES / f"err_confusion_pairs_{run}.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["true", "pred", "pixels"])
        w.writeheader()
        w.writerows(pairs[:20])

    # ---------------- 图
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot([r["bucket"] for r in boundary_rows], [r["miou"] for r in boundary_rows], "o-", color="#2f6db5")
    axes[0].set_title("mIoU vs distance to class boundary")
    axes[0].set_xlabel("distance bucket")
    axes[0].set_ylabel("mIoU")
    axes[0].grid(alpha=0.3)

    labs = [r["area_bucket"] for r in area_rows]
    vals = [r["recall"] or 0 for r in area_rows]
    bars = axes[1].bar(labs, vals, color="#2f9e6b")
    for b, v in zip(bars, vals):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    axes[1].set_title("object recall vs GT object area")
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / f"err_curves_{run}.png", dpi=150)
    plt.close(fig)

    # 最差若干张三联图
    per_image_err.sort(key=lambda t: -t[0])
    n = min(args.num_bad, len(per_image_err))
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.4 * n))
    axes = np.atleast_2d(axes)
    for k in range(n):
        err, name, raw, gt, pred = per_image_err[k]
        axes[k, 0].imshow(raw); axes[k, 0].set_title(f"{Path(name).name} errors={int(err)}", fontsize=8)
        axes[k, 1].imshow(colorize(gt.astype(np.uint8))); axes[k, 1].set_title("ground truth", fontsize=8)
        axes[k, 2].imshow(colorize(pred.astype(np.uint8))); axes[k, 2].set_title("prediction", fontsize=8)
        for ax in axes[k]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / f"err_worst_{run}.png", dpi=140)
    plt.close(fig)

    summary = {
        "run": run,
        "model": args.model,
        "ckpt": str(args.ckpt),
        "split": args.split,
        "n_images": len(items),
        "boundary": boundary_rows,
        "small_object": area_rows,
        "top_confusion_pairs": pairs[:10],
        "per_class_recall": (np.diag(cm) / cm.sum(1).clip(min=1)).round(4).tolist(),
    }
    (METRICS / f"err_{run}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[{run}] 边界带 mIoU:")
    for r in boundary_rows:
        print(f"  {r['bucket']:10s} mIoU={r['miou']:.4f}  像素占比 {r['pixels'] / sum(x['pixels'] for x in boundary_rows):5.1%}")
    print("\n小目标召回（GT 连通域，≥50% 像素对算召回）:")
    for r in area_rows:
        print(f"  {r['area_bucket']:14s} 物体 {r['objects']:5d}  召回 {r['recall']}")
    print("\nTop-5 混淆对:")
    for p in pairs[:5]:
        print(f"  {p['true']:12s} -> {p['pred']:12s} {p['pixels']:8d} 像素")
    print(f"\n输出: outputs/tables/err_*_{run}.csv, outputs/figures/err_curves_{run}.png, err_worst_{run}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
