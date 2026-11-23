"""M4/M5/M6：RemoteCLIP 在小块（patch）级别的对照实验。

为什么不按整图分类：LoveDA 是逐像素标注，整图标签只能用"众数类"，
实测 45.4% 的图众数类是背景、道路只有 0.4%，裸 OA 会被背景吃掉。
改成 224×224 小块（CLIP 原生输入尺寸），只保留单类纯度 ≥90% 的块。

三种实验：
  M4 零样本     用文本模板给 7 类打分，不训练
  M5 线性探针   冻结图像编码器提特征，在训练块上训 LogisticRegression
  M6 prompt     5 种模板，看文本描述对分数的影响

用法：
  python -m src.clip_eval --stage index      # 扫图，建 patch 索引（缓存）
  python -m src.clip_eval --stage all        # M4 + M5 + M6 + 检索 demo
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.loveda import CLASS_NAMES, CLASS_NAMES_CN  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "LoveDA"
TABLES = ROOT / "outputs" / "tables"
METRICS = ROOT / "outputs" / "metrics"
FIG = ROOT / "outputs" / "figures"
CACHE = ROOT / "data" / "patch_index.json"
WEIGHTS = ROOT / "models" / "RemoteCLIP-ViT-B-32.pt"

PATCH = 224
# 224 的块要铺满 1024，偏移取 0..800，步长 112（重叠一半），兼顾稀有类
GRID = list(range(0, 1024 - PATCH + 1, 112))
if GRID[-1] != 1024 - PATCH:
    GRID.append(1024 - PATCH)
PURITY = 0.85                 # 块内单类纯度阈值（90% 时道路只剩 37 块，太苛刻）
MAX_NODATA = 0.10
SEED = 3407

# 文本侧的类别描述（M6 里换模板，类别词不变）
PROMPT_CLASS = {
    "background": "background",
    "building": "building",
    "road": "road",
    "water": "water",
    "barren": "barren land",
    "forest": "forest",
    "agriculture": "farmland",
}

TEMPLATES = [
    "a satellite image of a {c}",
    "a satellite image of {c}",
    "an aerial image of a {c}",
    "a remote sensing image of a {c}",
    "a photo of a {c}",
]


# ---------------------------------------------------------------- patch index
def build_index() -> list[dict]:
    """扫 Train+Val，记录高纯度小块的坐标与标签。"""
    rng = random.Random(SEED)
    items: list[dict] = []
    for split in ("Train", "Val"):
        for dom in ("Urban", "Rural"):
            idir = RAW / split / dom / "images_png"
            mdir = RAW / split / dom / "masks_png"
            if not idir.is_dir():
                continue
            for p in tqdm(sorted(idir.glob("*.png")), desc=f"{split}/{dom}"):
                mp = mdir / p.name
                if not mp.exists():
                    continue
                with Image.open(mp) as mk:
                    m = np.asarray(mk.convert("L"))
                for y in GRID:
                    for x in GRID:
                        tile = m[y : y + PATCH, x : x + PATCH]
                        cnt = np.bincount(tile.ravel(), minlength=8)[:8]
                        tot = int(cnt.sum())
                        if tot == 0:
                            continue
                        nodata = cnt[0] / tot
                        if nodata > MAX_NODATA:
                            continue
                        body = cnt.copy()
                        body[0] = 0
                        top = int(body.argmax())
                        if body[top] / max(tot - cnt[0], 1) < PURITY:
                            continue
                        items.append(
                            {
                                "split": split,
                                "domain": dom,
                                "image": str(p.relative_to(RAW)).replace("\\", "/"),
                                "x": x,
                                "y": y,
                                "label": top - 1,  # 原始 1..7 -> 0..6
                            }
                        )
    rng.shuffle(items)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return items


def load_index() -> list[dict]:
    if not CACHE.exists():
        return build_index()
    return json.loads(CACHE.read_text(encoding="utf-8"))


def balanced_sample(items: list[dict], per_class: int, rng: random.Random) -> list[dict]:
    by = {c: [] for c in range(len(CLASS_NAMES))}
    for it in items:
        by[it["label"]].append(it)
    out = []
    for c, lst in by.items():
        rng.shuffle(lst)
        out += lst[:per_class]
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------- model / io
def load_clip():
    import open_clip

    if not WEIGHTS.exists():
        from huggingface_hub import hf_hub_download

        WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download("chendelong/RemoteCLIP", "RemoteCLIP-ViT-B-32.pt", local_dir=str(WEIGHTS.parent))
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=str(WEIGHTS))
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tokenizer


@torch.no_grad()
def encode_patches(model, preprocess, items: list[dict], device: str, bs: int = 128) -> np.ndarray:
    feats = []
    for i in tqdm(range(0, len(items), bs), desc="encode patches"):
        batch = items[i : i + bs]
        imgs = []
        for it in batch:
            with Image.open(RAW / it["image"]) as im:
                im = im.convert("RGB").crop((it["x"], it["y"], it["x"] + PATCH, it["y"] + PATCH))
                imgs.append(preprocess(im))
        x = torch.stack(imgs).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            f = model.encode_image(x)
        f = f.float()
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


@torch.no_grad()
def encode_texts(model, tokenizer, texts: list[str], device: str) -> np.ndarray:
    tok = tokenizer(texts).to(device)
    with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
        f = model.encode_text(tok)
    f = f.float()
    f = f / f.norm(dim=-1, keepdim=True)
    return f.cpu().numpy()


# ---------------------------------------------------------------- metrics
def report(y_true: np.ndarray, y_pred: np.ndarray, tag: str) -> dict:
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

    labels = list(range(len(CLASS_NAMES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_recall = (np.diag(cm) / cm.sum(axis=1).clip(min=1)).round(4).tolist()
    out = {
        "tag": tag,
        "n": int(len(y_true)),
        "oa": round(float((y_true == y_pred).mean()), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", labels=labels)), 4),
        "balanced_acc": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "per_class_recall": per_class_recall,
        "confusion": cm.tolist(),
        "classes": CLASS_NAMES,
    }
    return out


def print_report(r: dict) -> None:
    print(f"  {r['tag']:22s} OA={r['oa']:.4f}  macroF1={r['macro_f1']:.4f}  balAcc={r['balanced_acc']:.4f}")
    for name, rec in zip(CLASS_NAMES, r["per_class_recall"]):
        print(f"      {name:12s} recall={rec:.3f}")


def save_confusion_fig(r: dict, name: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(r["confusion"], dtype=float)
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title(f"{name}  (OA={r['oa']:.3f}, macroF1={r['macro_f1']:.3f})")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if cmn[i, j] > 0.01:
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="w" if cmn[i, j] > 0.5 else "k")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    out = FIG / f"clip_{name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- stages
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["index", "m4", "m5", "m6", "retrieval", "all"])
    ap.add_argument("--per-class-test", type=int, default=300)
    ap.add_argument("--per-class-train", type=int, default=600)
    args = ap.parse_args()

    TABLES.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(SEED)

    items = load_index() if args.stage in {"all", "m4", "m5", "m6", "retrieval"} else build_index()
    if args.stage == "index":
        from collections import Counter

        print("patch 总数:", len(items))
        print("按类:", {CLASS_NAMES[k]: v for k, v in sorted(Counter(i["label"] for i in items).items())})
        return 0

    from collections import Counter

    print(f"patch 索引 {len(items)} 个 | 按类 " + str({CLASS_NAMES[k]: v for k, v in sorted(Counter(i["label"] for i in items).items())}))

    tr = balanced_sample([i for i in items if i["split"] == "Train"], args.per_class_train, rng)
    te = balanced_sample([i for i in items if i["split"] == "Val"], args.per_class_test, rng)
    print(f"训练块 {len(tr)} / 测试块 {len(te)}（每类均衡）")

    model, preprocess, tokenizer = load_clip()
    model = model.to(device).eval()

    y_te = np.array([i["label"] for i in te])
    feat_te = encode_patches(model, preprocess, te, device)
    results: dict[str, dict] = {}

    if args.stage in {"all", "m4", "m5", "m6"}:
        # ---------------- M4 零样本
        cls_words = [PROMPT_CLASS[c] for c in CLASS_NAMES]
        texts = [TEMPLATES[0].format(c=w) for w in cls_words]
        tfeat = encode_texts(model, tokenizer, texts, device)
        logits = feat_te @ tfeat.T
        pred = logits.argmax(1)
        r = report(y_te, pred, "M4 zero-shot")
        results["M4"] = r
        print_report(r)
        save_confusion_fig(r, "m4_zeroshot")

        np.save(METRICS / "clip_text_emb.npy", tfeat)
        (METRICS / "clip_m4.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.stage in {"all", "m4", "m5", "m6"}:
        # ---------------- M6 prompt 模板
        rows = []
        for tpl in TEMPLATES:
            texts = [tpl.format(c=w) for w in cls_words]
            tf = encode_texts(model, tokenizer, texts, device)
            pr = (feat_te @ tf.T).argmax(1)
            rr = report(y_te, pr, tpl)
            rows.append({"template": tpl, "oa": rr["oa"], "macro_f1": rr["macro_f1"],
                         "balanced_acc": rr["balanced_acc"]})
            print(f"  M6 {tpl:38s} OA={rr['oa']:.4f} macroF1={rr['macro_f1']:.4f}")
        import csv

        with (TABLES / "clip_m6_prompts.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["template", "oa", "macro_f1", "balanced_acc"])
            w.writeheader()
            w.writerows(rows)
        (METRICS / "clip_m6.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        labels = [f"T{i+1}" for i in range(len(rows))]
        vals = [r["macro_f1"] for r in rows]
        bars = ax.bar(labels, vals, color="#2f6db5")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.set_ylabel("macro F1")
        ax.set_title("M6: prompt template sensitivity (RemoteCLIP zero-shot)")
        fig.tight_layout()
        fig.savefig(FIG / "clip_m6_prompts.png", dpi=150)
        plt.close(fig)

    if args.stage in {"all", "m5"}:
        # ---------------- M5 线性探针
        from sklearn.linear_model import LogisticRegression

        y_tr = np.array([i["label"] for i in tr])
        feat_tr = encode_patches(model, preprocess, tr, device)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
        clf.fit(feat_tr, y_tr)
        pred = clf.predict(feat_te)
        r = report(y_te, pred, "M5 linear probe")
        results["M5"] = r
        print_report(r)
        save_confusion_fig(r, "m5_probe")
        (METRICS / "clip_m5.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        np.savez(METRICS / "clip_features.npz", train=feat_tr, y_train=y_tr, test=feat_te, y_test=y_te)

    # ---------------- 汇总表
    if results:
        import csv

        rows = []
        for k, v in results.items():
            row = {"experiment": k, "OA": v["oa"], "macro_F1": v["macro_f1"], "balanced_acc": v["balanced_acc"]}
            for name, rec in zip(CLASS_NAMES, v["per_class_recall"]):
                row[f"recall_{name}"] = rec
            rows.append(row)
        keys = list(rows[0].keys())
        with (TABLES / "clip_m4_m5.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print("\n汇总 ->", TABLES / "clip_m4_m5.csv")
        for r in rows:
            print(f"  {r['experiment']:16s} OA={r['OA']:.4f} macroF1={r['macro_F1']:.4f} balAcc={r['balanced_acc']:.4f}")

    if args.stage in {"all", "retrieval"}:
        # ---------------- 检索 demo：每个类找 Top-5 最相似的测试块
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        texts = [TEMPLATES[0].format(c=PROMPT_CLASS[c]) for c in CLASS_NAMES]
        tf = encode_texts(model, tokenizer, texts, device)
        sims = feat_te @ tf.T
        fig, axes = plt.subplots(len(CLASS_NAMES), 5, figsize=(11, 2.1 * len(CLASS_NAMES)))
        for ci, cname in enumerate(CLASS_NAMES):
            order = np.argsort(-sims[:, ci])[:5]
            for j, idx in enumerate(order):
                it = te[int(idx)]
                with Image.open(RAW / it["image"]) as im:
                    crop = im.convert("RGB").crop((it["x"], it["y"], it["x"] + PATCH, it["y"] + PATCH))
                ax = axes[ci, j]
                ax.imshow(crop)
                ax.set_title(f"sim={sims[int(idx), ci]:.3f}\ntrue={CLASS_NAMES[it['label']]}", fontsize=7)
                ax.axis("off")
            axes[ci, 0].set_ylabel(cname, fontsize=9)
        fig.suptitle("RemoteCLIP retrieval: top-5 patches per class query", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(FIG / "clip_retrieval.png", dpi=130)
        plt.close(fig)
        print("检索图 ->", FIG / "clip_retrieval.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
