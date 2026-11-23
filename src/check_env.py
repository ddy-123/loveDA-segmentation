"""环境验收：一次性核对 torch/CUDA/GPU/关键依赖，并把环境事实落盘。

对应方案文档第 2.2 节验收命令，外加逐算子体检（避免"能 import 但算不出"）。
用法：python -m src.check_env
输出：outputs/tables/env.json，同时在 stdout 打印表格。
"""
from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tables"

PACKAGES = [
    "torch",
    "torchvision",
    "numpy",
    "transformers",
    "timm",
    "segmentation-models-pytorch",
    "opencv-python",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "pillow",
    "tqdm",
    "pyyaml",
    "tensorboard",
    "onnx",
    "onnxruntime-gpu",
    "huggingface-hub",
]


def pkg_versions() -> dict[str, str]:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "MISSING"
    return out


def gpu_facts() -> dict:
    import torch

    facts: dict = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "is_available": torch.cuda.is_available(),
        "bf16_supported": None,
    }
    if not torch.cuda.is_available():
        return facts
    props = torch.cuda.get_device_properties(0)
    facts.update(
        {
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability()),
            "total_mem_gb": round(props.total_memory / 1024**3, 2),
            "sm_count": props.multi_processor_count,
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "cuda_version_runtime": torch.version.cuda,
        }
    )
    return facts


def op_battery() -> dict:
    """跑分割任务真实用到的算子；sm_120 上没有 kernel 会在这里炸出来。"""
    import torch
    import torch.nn.functional as F

    res: dict[str, str] = {}
    if not torch.cuda.is_available():
        return {"skipped": "cuda not available"}

    x = torch.randn(2, 3, 512, 512, device="cuda")
    conv = torch.nn.Conv2d(3, 32, 3, padding=1).cuda()

    def run(name, fn):
        try:
            torch.cuda.synchronize()
            t0 = time.time()
            fn()
            torch.cuda.synchronize()
            res[name] = f"ok {(time.time() - t0) * 1000:.0f}ms"
        except Exception as e:  # noqa: BLE001
            res[name] = "FAIL " + str(e).splitlines()[0][:120]

    run("conv2d", lambda: conv(x))
    run("conv2d_bwd", lambda: conv(x).mean().backward())
    run("amp_fp16", lambda: F.conv2d(x.half(), conv.weight.half(), padding=1))
    run("interpolate", lambda: F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
    run("softmax", lambda: torch.softmax(x, dim=1))
    t = torch.randn(2, 197, 768, device="cuda")
    run("sdpa", lambda: F.scaled_dot_product_attention(t, t, t))
    run("cross_entropy", lambda: F.cross_entropy(x, torch.randint(0, 3, (2, 512, 512), device="cuda")))
    return res


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    facts = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "python_exe": sys.executable,
        "platform": platform.platform(),
        "packages": pkg_versions(),
    }
    facts.update(gpu_facts())
    facts["ops"] = op_battery()

    print(f"python      : {facts['python']}  ({facts['python_exe']})")
    print(f"platform    : {facts['platform']}")
    print(f"torch       : {facts['torch']}  cuda_build={facts['cuda_build']}  cudnn={facts['cudnn']}")
    if facts.get("is_available"):
        print(
            f"gpu         : {facts['device']}  {facts['total_mem_gb']} GB  "
            f"cap={tuple(facts['capability'])}  sm={facts['sm_count']}  bf16={facts['bf16_supported']}"
        )
    print(f"cuda_available: {facts['is_available']}")
    print("\n--- 依赖 ---")
    for k, v in facts["packages"].items():
        flag = "  <-- MISSING" if v == "MISSING" else ""
        print(f"  {k:30s} {v}{flag}")
    print("\n--- 算子体检 ---")
    for k, v in facts["ops"].items():
        print(f"  {k:16s} {v}")

    ok = bool(facts.get("is_available")) and not any(str(v).startswith("FAIL") for v in facts["ops"].values())
    missing = [k for k, v in facts["packages"].items() if v == "MISSING"]
    facts["ok"] = ok and not missing

    path = OUT / "env.json"
    path.write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nenv -> {path}")
    print(f"verdict: {'PASS' if facts['ok'] else 'FAIL'}" + (f"  缺包: {missing}" if missing else ""))
    return 0 if facts["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
