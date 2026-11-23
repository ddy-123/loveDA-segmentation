"""校验 / 补下 LoveDA 数据集。

数据来源（2026-11-17 定）：
  - 官方完整版需自行下载并放置于 `data/raw/LoveDA`（本仓库不包含数据）
  - 换机器需要重下时：zenodo DNS 不通、网盘不可脚本化，用 HF 镜像
    `ahsennazir/loveDA`（与本目录 Train/Val 逐字节一致，已抽样 md5 验证）。

镜像下载的三个坑（都踩过）：
  - hf_xet 报 "I/O error: 拒绝访问 (os error 5)" → HF_HUB_DISABLE_XET=1
  - hf-mirror 高并发返回 429 → 4 并发 + 断点续传
  - 用户目录 .cache 不可写 → HF_HOME / TORCH_HOME 指到项目内

用法：
  python -m src.download            # 校验本地数据 + 写 manifest
  python -m src.download --fetch    # 本地没有时从 HF 镜像下载
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ID = "ahsennazir/loveDA"
DEFAULT_ENDPOINT = "https://hf-mirror.com"

# 官方张数
EXPECTED_IMAGES = {"Train": 2522, "Val": 1669, "Test": 1796}
EXPECTED_BY_CELL = {
    ("Train", "Urban"): 1156,
    ("Train", "Rural"): 1366,
    ("Val", "Urban"): 677,
    ("Val", "Rural"): 992,
    ("Test", "Urban"): 820,
    ("Test", "Rural"): 976,
}

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "LoveDA"
MANIFEST = ROOT / "data" / "manifest.json"


def download(endpoint: str = DEFAULT_ENDPOINT, workers: int = 4, retries: int = 12) -> None:
    os.environ["HF_ENDPOINT"] = endpoint
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import snapshot_download

    print(f"[download] repo={REPO_ID} endpoint={endpoint} workers={workers} -> {RAW}")
    for attempt in range(1, retries + 1):
        try:
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                local_dir=str(RAW),
                allow_patterns=["Train/*", "Val/*"],
                max_workers=workers,
            )
            print(f"[download] done on attempt {attempt}")
            return
        except Exception as e:  # noqa: BLE001
            wait = min(60, 5 * attempt)
            print(f"[download] attempt {attempt}/{retries} failed: {str(e).splitlines()[0][:200]}")
            print(f"[download] sleeping {wait}s then resuming ...", flush=True)
            time.sleep(wait)
    raise SystemExit("下载多次失败，请稍后重试（已下载部分会保留）")


def scan() -> dict:
    """扫描本地目录，返回计数与异常。"""
    counts: Counter = Counter()
    missing_masks: list[str] = []
    size_anomalies: list[dict] = []
    total_bytes = 0
    for split in ("Train", "Val", "Test"):
        for domain in ("Urban", "Rural"):
            img_dir = RAW / split / domain / "images_png"
            msk_dir = RAW / split / domain / "masks_png"
            if not img_dir.is_dir():
                continue
            imgs = sorted(p.name for p in img_dir.glob("*.png"))
            msks = {p.name for p in msk_dir.glob("*.png")} if msk_dir.is_dir() else set()
            counts[f"{split}/{domain}/images_png"] = len(imgs)
            counts[f"{split}/{domain}/masks_png"] = len(msks)
            total_bytes += sum(p.stat().st_size for p in img_dir.glob("*.png"))
            if msk_dir.is_dir():
                total_bytes += sum(p.stat().st_size for p in msk_dir.glob("*.png"))
            if not msks:
                continue  # Test 无标注，不比对
            for name in imgs:
                if name not in msks:
                    missing_masks.append(f"{split}/{domain}/images_png/{name}")
            for p in list(img_dir.glob("*.png"))[:2000]:
                try:
                    from PIL import Image

                    with Image.open(p) as im:
                        if im.size != (1024, 1024) or im.mode != "RGB":
                            size_anomalies.append(
                                {"path": str(p.relative_to(RAW)), "size": im.size, "mode": im.mode}
                            )
                except Exception as e:  # noqa: BLE001
                    size_anomalies.append({"path": str(p.relative_to(RAW)), "error": repr(e)})
    return {
        "counts": dict(counts),
        "missing_masks": missing_masks,
        "size_anomalies": size_anomalies,
        "total_bytes": total_bytes,
    }


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sample_md5() -> list[dict]:
    out = []
    for split in ("Train", "Val"):
        for domain in ("Urban", "Rural"):
            imgs = sorted((RAW / split / domain / "images_png").glob("*.png"))
            msks = sorted((RAW / split / domain / "masks_png").glob("*.png"))
            if not imgs:
                continue
            for p in (imgs[0], msks[0]) if msks else (imgs[0],):
                out.append({"path": str(p.relative_to(RAW)), "bytes": p.stat().st_size, "md5": md5_of(p)})
    return out


def verify(report: dict) -> tuple[bool, list[str]]:
    ok = True
    msgs: list[str] = []
    for split, expected in EXPECTED_IMAGES.items():
        got = sum(
            v for k, v in report["counts"].items() if k.startswith(split + "/") and k.endswith("images_png")
        )
        flag = "OK" if got == expected else "MISMATCH"
        if got != expected:
            ok = False
        msgs.append(f"{split} images: {got} (expect {expected}) [{flag}]")
    for (split, domain), expected in EXPECTED_BY_CELL.items():
        got = report["counts"].get(f"{split}/{domain}/images_png", 0)
        if got != expected:
            ok = False
            msgs.append(f"{split}/{domain}: {got} (expect {expected}) [MISMATCH]")
    for k, v in report["counts"].items():
        if not k.endswith("images_png"):
            continue
        if k.startswith("Test/"):
            continue  # Test 本来就没有标注
        m = report["counts"].get(k.replace("images_png", "masks_png"), 0)
        if m != v:
            ok = False
            msgs.append(f"{k}: images {v} != masks {m} [MISMATCH]")
    if report["missing_masks"]:
        ok = False
        msgs.append(f"缺 mask 的图: {len(report['missing_masks'])} 张")
    if report["size_anomalies"]:
        ok = False
        msgs.append(f"尺寸/通道异常: {len(report['size_anomalies'])} 个")
    return ok, msgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="本地没有数据时从 HF 镜像下载")
    ap.add_argument("--check-only", action="store_true", help="（兼容旧用法，默认就是只校验）")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if args.fetch:
        download(args.endpoint, args.workers)

    report = scan()
    ok, msgs = verify(report)
    print("\n".join(msgs))
    for k in sorted(report["counts"]):
        print(f"  {k:32s} {report['counts'][k]}")

    test_dir = RAW / "Test"
    test_images = sum(v for k, v in report["counts"].items() if k.startswith("Test/") and k.endswith("images_png"))
    test_masks = sum(v for k, v in report["counts"].items() if k.startswith("Test/") and k.endswith("masks_png"))
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "root": str(RAW),
        "is_link": RAW.is_symlink() or bool(getattr(RAW, "is_junction", lambda: False)()),
        "source": {"local_official": "D:/study/数据智能/语义分割/2021LoveDA", "mirror": REPO_ID},
        "counts": report["counts"],
        "total_bytes": report["total_bytes"],
        "missing_masks": report["missing_masks"][:20],
        "size_anomalies": report["size_anomalies"][:20],
        "samples_md5": sample_md5(),
        "test_present": test_dir.is_dir(),
        "test_images": test_images,
        "test_has_masks": test_masks > 0,
        "ok": ok,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nmanifest -> {MANIFEST}   verify_ok={ok}")
    print(f"Test: {test_images} 张影像, masks={test_masks} -> 官方不公开标注，只能做无监督推理/可视化")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
