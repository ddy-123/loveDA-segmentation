"""导出 ONNX 并测速：PyTorch vs onnxruntime（CPU / CUDA）。

用法：
  python -m src.export_onnx --model segformer_b2 --ckpt outputs/ckpt/segformer_b2/best.pth
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.models import build_model

ROOT = Path(__file__).resolve().parents[1]
ONNX_DIR = ROOT / "outputs" / "onnx"
TABLES = ROOT / "outputs" / "tables"
METRICS = ROOT / "outputs" / "metrics"


def bench_torch(model, x, n: int = 30) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            model(x)
        torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000


def bench_ort(sess, np_in, n: int = 30) -> float:
    name = sess.get_inputs()[0].name
    for _ in range(5):
        sess.run(None, {name: np_in})
    t0 = time.time()
    for _ in range(n):
        sess.run(None, {name: np_in})
    return (time.time() - t0) / n * 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dynamic", action="store_true", help="导出动态 H/W（推理大图用）")
    args = ap.parse_args()

    import onnx
    import onnxruntime as ort

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = Path(args.ckpt).parent.name

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = state.get("cfg") or {}
    model = build_model(args.model, pretrained=False, arch=cfg.get("pretrained"))
    model.load_state_dict(state.get("model", state), strict=True)
    model = model.to(device).eval()

    x = torch.randn(1, 3, args.size, args.size, device=device)
    out_path = ONNX_DIR / f"{tag}_{args.size}{'_dyn' if args.dynamic else ''}.onnx"
    dynamic_axes = {"input": {2: "h", 3: "w"}, "output": {2: "h", 3: "w"}} if args.dynamic else None
    torch.onnx.export(
        model, x, str(out_path),
        input_names=["input"], output_names=["output"],
        opset_version=args.opset, dynamic_axes=dynamic_axes, dynamo=False,
    )
    onnx.checker.check_model(str(out_path))
    mb = out_path.stat().st_size / 1024**2
    print(f"[export] {out_path.name}  {mb:.1f} MB  opset={args.opset} dynamic={args.dynamic}")

    # 数值一致性
    with torch.no_grad():
        ref = model(x).float().cpu().numpy()
    np_in = x.detach().cpu().numpy()

    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(out_path), providers=providers)
    got = sess.run(None, {sess.get_inputs()[0].name: np_in})[0]
    max_diff = float(np.abs(ref - got).max())
    print(f"[verify] max|torch - ort| = {max_diff:.2e}  providers={sess.get_providers()}")

    # 测速
    ms_torch = bench_torch(model, x)
    ms_ort_gpu = bench_ort(sess, np_in) if "CUDAExecutionProvider" in sess.get_providers() else None
    sess_cpu = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ms_ort_cpu = bench_ort(sess_cpu, np_in)

    rows = [
        {"impl": "pytorch+cuda", "ms_per_512": round(ms_torch, 2), "img_per_s": round(1000 / ms_torch, 1)},
        {"impl": "onnxruntime+cpu", "ms_per_512": round(ms_ort_cpu, 2), "img_per_s": round(1000 / ms_ort_cpu, 1)},
    ]
    if ms_ort_gpu:
        rows.append({"impl": "onnxruntime+cuda", "ms_per_512": round(ms_ort_gpu, 2),
                     "img_per_s": round(1000 / ms_ort_gpu, 1)})

    import csv

    TABLES.mkdir(parents=True, exist_ok=True)
    with (TABLES / f"onnx_speed_{tag}.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["impl", "ms_per_512", "img_per_s"])
        w.writeheader()
        w.writerows(rows)

    info = {
        "model": args.model,
        "ckpt": str(args.ckpt),
        "onnx": str(out_path),
        "size_mb": round(mb, 1),
        "opset": args.opset,
        "dynamic": args.dynamic,
        "max_abs_diff": max_diff,
        "speed": rows,
    }
    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / f"onnx_{tag}.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n速度（单张 512×512，batch=1）:")
    for r in rows:
        print(f"  {r['impl']:20s} {r['ms_per_512']:7.2f} ms  {r['img_per_s']:6.1f} img/s")
    print(f"\n-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
