"""评估：512 滑窗推理 + mIoU / Kappa / 混淆矩阵 / 每类指标 / 可视化。

统一协议（方案文档 4.3）：512 窗口、stride 256，拼回 1024 全图后再算指标。
三个模型必须用同一套，否则数字不可比。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.dataset import LoveDADataset, read_split
from src.loveda import CLASS_NAMES, CLASS_NAMES_CN, IGNORE_INDEX, PALETTE, RAW_TO_IDX
from src.models import build_model

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "LoveDA"
TABLES = ROOT / "outputs" / "tables"
METRICS = ROOT / "outputs" / "metrics"
FIG = ROOT / "outputs" / "figures"


@torch.no_grad()
def sliding_infer(model, x: torch.Tensor, crop: int = 512, stride: int = 256, n_class: int = 7) -> torch.Tensor:
    """x: (1,3,H,W) -> logits (1,n_class,H,W)。窗口重叠区取平均。"""
    _, _, h, w = x.shape
    logits = torch.zeros((1, n_class, h, w), device=x.device, dtype=torch.float32)
    weight = torch.zeros((1, 1, h, w), device=x.device, dtype=torch.float32)
    ys = list(range(0, max(h - crop, 0) + 1, stride))
    xs = list(range(0, max(w - crop, 0) + 1, stride))
    if ys[-1] != h - crop:
        ys.append(h - crop)
    if xs[-1] != w - crop:
        xs.append(w - crop)
    for y in ys:
        for x0 in xs:
            patch = x[:, :, y : y + crop, x0 : x0 + crop]
            out = model(patch)
            logits[:, :, y : y + crop, x0 : x0 + crop] += out
            weight[:, :, y : y + crop, x0 : x0 + crop] += 1.0
    return logits / weight.clamp_min(1.0)


class ConfusionMatrix:
    def __init__(self, n: int = 7) -> None:
        self.n = n
        self.mat = np.zeros((n, n), dtype=np.int64)

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        m = gt != IGNORE_INDEX
        idx = gt[m].astype(np.int64) * self.n + pred[m].astype(np.int64)
        self.mat += np.bincount(idx, minlength=self.n**2).reshape(self.n, self.n)

    def metrics(self) -> dict:
        m = self.mat
        tp = np.diag(m).astype(np.float64)
        fp = m.sum(0) - tp
        fn = m.sum(1) - tp
        denom = tp + fp + fn
        iou = np.divide(tp, denom, out=np.full_like(tp, np.nan), where=denom > 0)
        f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.full_like(tp, np.nan), where=2 * tp + fp + fn > 0)
        total = m.sum()
        po = tp.sum() / total if total else float("nan")
        pe = (m.sum(0) * m.sum(1)).sum() / (total**2) if total else float("nan")
        kappa = (po - pe) / (1 - pe) if total and (1 - pe) else float("nan")
        return {
            "miou": float(np.nanmean(iou)),
            "miou_no_bg": float(np.nanmean(iou[1:])),
            "pixel_acc": float(po),
            "kappa": float(kappa),
            "per_class_iou": [float(v) for v in iou],
            "per_class_f1": [float(v) for v in f1],
            "confusion": m.tolist(),
        }


def colorize(idx: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*idx.shape, 3), dtype=np.uint8)
    for raw, i in RAW_TO_IDX.items():
        rgb[idx == i] = PALETTE[i]
    return rgb


def visualize(img: np.ndarray, gt: np.ndarray, pred: np.ndarray, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].imshow(img)
    axes[0].set_title("image")
    axes[1].imshow(colorize(gt))
    axes[1].set_title("ground truth")
    axes[2].imshow(colorize(pred))
    axes[2].set_title("prediction")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="unet / deeplabv3plus / segformer_b2")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="dev", help="dev / test / val 或 splits 下的文件名（不带 .txt）")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只评估前 N 张（调试用）")
    ap.add_argument("--tag", default="", help="结果目录后缀，便于区分实验")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    saved_cfg = state.get("cfg") or {}
    arch = saved_cfg.get("pretrained") or None          # 从 ckpt 里读回结构信息（B0/B2 差别就在这）
    model_name = args.model or saved_cfg.get("model")
    model = build_model(model_name, pretrained=False, arch=arch).to(device)
    model.load_state_dict(state.get("model", state), strict=True)
    model.eval()
    print(f"[eval] {model_name} arch={arch or 'default'} step={state.get('step')} "
          f"quick={state.get('quick_miou')}", flush=True)

    items = read_split(ROOT / "data" / "splits" / f"{args.split}.txt")
    if args.limit:
        items = items[: args.limit]
    ds = LoveDADataset(items, crop=None, train=False)
    cm = ConfusionMatrix(len(CLASS_NAMES))

    run = f"{args.model}_{args.split}" + (f"_{args.tag}" if args.tag else "")
    (FIG / run).mkdir(parents=True, exist_ok=True)

    times = []
    for i in tqdm(range(len(ds)), desc=f"eval {run}"):
        s = ds[i]
        x = s["image"].unsqueeze(0).to(device)
        torch.cuda.synchronize()
        import time

        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            out = sliding_infer(model, x, args.crop, args.stride)
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)
        pred = out.argmax(1)[0].cpu().numpy().astype(np.uint8)
        gt = s["mask"].numpy().astype(np.uint8)
        cm.update(pred, gt)
        if i < 6:
            img = (s["image"].numpy().transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            visualize(img, gt, pred, FIG / run / f"{Path(s['name']).stem}.png")

    m = cm.metrics()
    m.update(
        {
            "model": args.model,
            "split": args.split,
            "ckpt": str(args.ckpt),
            "n_images": len(items),
            "ms_per_image": float(np.mean(times)) if times else None,
            "class_names": CLASS_NAMES,
        }
    )
    METRICS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    (METRICS / f"eval_{run}.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = [
        {
            "model": args.model,
            "split": args.split,
            "class": CLASS_NAMES_CN[i] + "/" + CLASS_NAMES[i],
            "iou": m["per_class_iou"][i],
            "f1": m["per_class_f1"][i],
        }
        for i in range(len(CLASS_NAMES))
    ]
    import csv

    with (TABLES / f"eval_{run}_per_class.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["model", "split", "class", "iou", "f1"])
        w.writeheader()
        w.writerows(rows)

    print(f"mIoU={m['miou']:.4f}  mIoU(no bg)={m['miou_no_bg']:.4f}  "
          f"pixel_acc={m['pixel_acc']:.4f}  kappa={m['kappa']:.4f}  {m['ms_per_image']:.0f} ms/img")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
