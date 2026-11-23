"""训练主循环：AMP、梯度累积、poly/cosine 调度、warmup、断点续训、早停、TensorBoard。

用法：
  python -m src.train --config configs/segformer_b2.yaml
  python -m src.train --config configs/unet.yaml --iters 200 --tag smoke
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset import LoveDADataset, read_split
from src.loveda import CLASS_NAMES, IGNORE_INDEX
from src.models import build_model, count_params

ROOT = Path(__file__).resolve().parents[1]


def load_cfg(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    tr = cfg["train"]
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or "bias" in name or "norm" in name.lower() else decay).append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": tr["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=tr["lr"],
    )


def lr_at(step: int, cfg: dict) -> float:
    tr = cfg["train"]
    total, warm, base = tr["iters"], tr["warmup_iters"], tr["lr"]
    if step < warm:
        return base * (step + 1) / max(warm, 1)
    if tr["scheduler"] == "poly":
        p = (1 - (step - warm) / max(total - warm, 1)) ** tr.get("power", 1.0)
    elif tr["scheduler"] == "cosine":
        p = 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))
    else:
        p = 1.0
    return base * max(p, 0.0)


@torch.no_grad()
def evaluate_quick(model: nn.Module, loader: DataLoader, device: str, max_batches: int = 20,
                   crop: int = 512) -> float:
    """训练中的快速验证：中心裁 crop×crop，只为看趋势，不作最终数字。

    必须和训练分辨率一致：训练时是 512 裁剪，若这里直接喂 1024 全图，
    BN 统计和位置编码都会偏，指标会失真（也慢一倍多）。
    """
    model.eval()
    inter = np.zeros(len(CLASS_NAMES))
    union = np.zeros(len(CLASS_NAMES))
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        h, w = x.shape[-2:]
        if crop and (h > crop or w > crop):
            top = max((h - crop) // 2, 0)
            left = max((w - crop) // 2, 0)
            x = x[:, :, top : top + crop, left : left + crop]
            y = y[:, top : top + crop, left : left + crop]
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            out = model(x)
        pred = out.argmax(1)
        for c in range(len(CLASS_NAMES)):
            m = y == c
            inter[c] += ((pred == c) & m).sum().item()
            union[c] += ((pred == c) | m).sum().item()
    model.train()
    iou = np.divide(inter, union, out=np.full_like(inter, np.nan), where=union > 0)
    return float(np.nanmean(iou))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--iters", type=int, default=0, help="覆盖配置里的 iters（调试用）")
    ap.add_argument("--tag", default="", help="实验后缀，输出目录用")
    ap.add_argument("--resume", default="", help="checkpoint 路径")
    ap.add_argument("--reset-best", action="store_true", help="续训但重置 best（换了验证口径时用）")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="dev")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.iters:
        cfg["train"]["iters"] = args.iters
    seed = cfg["train"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = cfg["name"] + (f"_{args.tag}" if args.tag else "")
    ckpt_dir = ROOT / "outputs" / "ckpt" / run
    log_dir = ROOT / "outputs" / "logs" / run
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    splits = ROOT / "data" / "splits"
    train_items = read_split(splits / f"{args.train_split}.txt")
    val_items = read_split(splits / f"{args.val_split}.txt")

    tr = cfg["train"]
    ds_tr = LoveDADataset(
        train_items,
        crop=tr["crop"],
        train=True,
        use_color=bool(cfg["aug"].get("use_color")),
        label_shift=cfg.get("label_shift", True),
    )
    # 快速验证用 600 张（原先 400 张 × 20 个 batch 实际只看了 160 张，波动太大）
    ds_va = LoveDADataset(val_items[:600], crop=tr["crop"], train=False, label_shift=cfg.get("label_shift", True))
    dl_tr = DataLoader(
        ds_tr,
        batch_size=tr["batch"],
        shuffle=True,
        num_workers=tr["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=tr["num_workers"] > 0,
    )
    dl_va = DataLoader(ds_va, batch_size=tr["batch"], shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(cfg["model"], pretrained=bool(cfg.get("pretrained"))).to(device)
    model.train()
    opt = make_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=tr["amp"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    start_step, best, bad = 0, -1.0, 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_step = state.get("step", 0)
        best = state.get("best", -1.0)
        if args.reset_best:
            best, bad = -1.0, 0
        print(f"[resume] from {args.resume} @ step {start_step}  best={best}", flush=True)

    writer = SummaryWriter(str(log_dir))
    print(f"[cfg] {run} model={cfg['model']} iters={tr['iters']} batch={tr['batch']} accum={tr['accum']} "
          f"lr={tr['lr']} amp={tr['amp']} params={count_params(model)} device={device}")
    print(f"[data] train={len(ds_tr)} val={len(ds_va)}")

    it = iter(dl_tr)
    pbar = tqdm(range(start_step, tr["iters"]), initial=start_step, total=tr["iters"], desc=run, ncols=100)
    t_last = time.time()
    for step in pbar:
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(tr["accum"]):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl_tr)
                batch = next(it)
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=tr["amp"] and device == "cuda"):
                out = model(x)
                loss = loss_fn(out, y) / tr["accum"]
            scaler.scale(loss).backward()
            loss_sum += loss.item()
        scaler.step(opt)
        scaler.update()

        if step % 50 == 0:
            writer.add_scalar("train/loss", loss_sum, step)
            writer.add_scalar("train/lr", lr, step)
        pbar.set_postfix(loss=f"{loss_sum:.3f}", lr=f"{lr:.1e}")

        if (step + 1) % tr.get("val_interval", 2000) == 0 or step + 1 == tr["iters"]:
            miou = evaluate_quick(model, dl_va, device, max_batches=max(10, len(ds_va) // tr["batch"]),
                                  crop=tr["crop"])
            writer.add_scalar("val/miou_quick", miou, step)
            print(f"\n[step {step+1}] quick mIoU(dev 子集, 非滑窗) = {miou:.4f}  loss={loss_sum:.3f}", flush=True)
            improved = miou > best
            if improved:
                best = miou
                bad = 0
            else:
                bad += 1
            # 注意：best 要在更新之后再写进 state，否则 best.pth 里记的是上一轮的旧值
            state = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step + 1,
                     "best": best, "quick_miou": miou, "cfg": cfg, "kind": "downstream"}
            torch.save(state, ckpt_dir / "last.pth")
            if improved:
                torch.save(state, ckpt_dir / "best.pth")
            elif bad >= cfg["eval"]["early_stop_patience"]:
                print(f"[early stop] 连续 {bad} 次未提升，停在第 {step+1} 步", flush=True)
                break
        t_last = time.time()

    writer.close()
    (log_dir / "summary.json").write_text(
        json.dumps({"run": run, "best_quick_miou": best, "iters": tr["iters"], "cfg": cfg},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] best quick mIoU={best:.4f}  ckpt -> {ckpt_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
