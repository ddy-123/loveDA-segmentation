"""数据质量分析：类别像素占比、no-data 比例、尺寸异常、重复图、跨域色调。

输出（全部可追溯，报告数字只从这里取）：
  outputs/tables/class_pixel_ratio.csv   每 split/domain 的原始类别像素占比
  outputs/tables/image_level_stats.csv   每张图一行：no-data 比例、出现的类、亮度/饱和度
  outputs/tables/domain_stats.csv        每 split/domain 汇总
  outputs/tables/duplicates.csv          完全重复的文件
  outputs/metrics/quality.json           汇总（含 6 项质量指标）
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "LoveDA"
TABLES = ROOT / "outputs" / "tables"
METRICS = ROOT / "outputs" / "metrics"

RAW_VALUES = list(range(8))  # 0..7


def iter_pairs(splits=("Train", "Val"), domains=("Urban", "Rural")):
    for split in splits:
        for domain in domains:
            img_dir = RAW / split / domain / "images_png"
            msk_dir = RAW / split / domain / "masks_png"
            if not img_dir.is_dir() or not msk_dir.is_dir():
                continue
            for img in sorted(img_dir.glob("*.png")):
                msk = msk_dir / img.name
                if not msk.exists():
                    continue  # 下载未完成时跳过，保证可重复运行
                yield split, domain, img, msk


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["Train", "Val"])
    ap.add_argument("--no-hash", action="store_true", help="跳过重复图检测（省时间）")
    args = ap.parse_args()

    TABLES.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)

    pixel_counts: dict[tuple[str, str], np.ndarray] = defaultdict(lambda: np.zeros(8, dtype=np.int64))
    rows: list[dict] = []
    hashes: dict[str, list[str]] = defaultdict(list)
    size_bad: list[dict] = []
    mask_vals_seen: Counter = Counter()

    pairs = list(iter_pairs(tuple(args.splits)))
    print(f"扫描 {len(pairs)} 对 image/mask ...")
    for split, domain, img_path, msk_path in tqdm(pairs):
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            if im.size != (1024, 1024):
                size_bad.append({"path": str(img_path.relative_to(RAW)), "size": im.size})
            arr = np.asarray(im, dtype=np.float32)
        with Image.open(msk_path) as mk:
            m = np.asarray(mk.convert("L"), dtype=np.uint8)
        if m.shape != (1024, 1024):
            size_bad.append({"path": str(msk_path.relative_to(RAW)), "size": m.shape})

        cnt = np.bincount(m.ravel(), minlength=8)[:8]
        pixel_counts[(split, domain)] += cnt
        mask_vals_seen.update({int(v): int(c) for v, c in enumerate(cnt) if c})

        present = [i for i in range(1, 8) if cnt[i] > 0]
        nodata = cnt[0] / max(int(cnt.sum()), 1)
        hsv = np.asarray(Image.fromarray(arr.astype(np.uint8)).convert("HSV"), dtype=np.float32)
        rows.append(
            {
                "split": split,
                "domain": domain,
                "name": img_path.name,
                "nodata_ratio": nodata,
                "n_classes_present": len(present),
                "brightness": float(arr.mean()),
                "saturation": float(hsv[..., 1].mean()),
                "hue_mean": float(hsv[..., 0].mean()),
            }
        )
        if not args.no_hash:
            hashes[md5(img_path)].append(str(img_path.relative_to(RAW)))

    # 1) 类别像素占比
    ratios = []
    for (split, domain), cnt in sorted(pixel_counts.items()):
        total = int(cnt.sum())
        row = {"split": split, "domain": domain, "total_pixels": total}
        for v in RAW_VALUES:
            row[f"raw{v}_ratio"] = float(cnt[v] / total) if total else 0.0
        ratios.append(row)
    class_df = pd.DataFrame(ratios)
    class_df.to_csv(TABLES / "class_pixel_ratio.csv", index=False, encoding="utf-8-sig")

    img_df = pd.DataFrame(rows)
    img_df.to_csv(TABLES / "image_level_stats.csv", index=False, encoding="utf-8-sig")

    # 2) 每 split/domain 汇总
    dom = (
        img_df.groupby(["split", "domain"])
        .agg(
            n=("name", "size"),
            nodata_ratio_mean=("nodata_ratio", "mean"),
            nodata_ratio_max=("nodata_ratio", "max"),
            n_classes_mean=("n_classes_present", "mean"),
            brightness_mean=("brightness", "mean"),
            brightness_std=("brightness", "std"),
            saturation_mean=("saturation", "mean"),
            hue_mean=("hue_mean", "mean"),
        )
        .reset_index()
    )
    dom.to_csv(TABLES / "domain_stats.csv", index=False, encoding="utf-8-sig")

    # 3) 重复图
    dups = []
    for h, paths in hashes.items():
        if len(paths) > 1:
            dups.append({"md5": h, "n": len(paths), "paths": "|".join(paths)})
    pd.DataFrame(dups).to_csv(TABLES / "duplicates.csv", index=False, encoding="utf-8-sig")

    summary = {
        "n_pairs": int(len(img_df)),
        "raw_values_seen": sorted(int(v) for v in mask_vals_seen),
        "nodata_ratio_overall": float(img_df["nodata_ratio"].mean()),
        "nodata_ratio_p95": float(img_df["nodata_ratio"].quantile(0.95)),
        "n_duplicates": int(len(dups)),
        "n_size_anomalies": int(len(size_bad)),
        "size_anomalies": size_bad[:20],
        "per_image_n_classes_hist": img_df["n_classes_present"].value_counts().sort_index().to_dict(),
        "class_ratio_by_domain": class_df.to_dict(orient="records"),
        "domain_stats": dom.to_dict(orient="records"),
    }
    (METRICS / "quality.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 原始 mask 取值 === ", summary["raw_values_seen"])
    print("=== 类别像素占比（按 split/domain） ===")
    print(class_df.round(4).to_string(index=False))
    print(f"\nno-data 比例均值 {summary['nodata_ratio_overall']:.4f} / p95 {summary['nodata_ratio_p95']:.4f}")
    print(f"重复图 {len(dups)} 组 / 尺寸异常 {len(size_bad)} 个")
    print("\n=== 域统计 ===")
    print(dom.round(3).to_string(index=False))
    print(f"\n输出目录: {TABLES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
