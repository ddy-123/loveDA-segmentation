"""LoveDA Dataset：读取、标签重映射、训练/验证增强。

split 文件格式：每行一个相对路径（相对 data/raw/LoveDA），指向 images_png 下的图，
mask 由路径推导（images_png -> masks_png）。
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.loveda import IGNORE_INDEX, remap_mask

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "LoveDA"


def read_split(path: Path) -> list[str]:
    return [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]


class LoveDADataset(Dataset):
    def __init__(
        self,
        items: list[str],
        root: Path = RAW,
        crop: int | None = 512,
        train: bool = True,
        use_color: bool = False,
        label_shift: bool = True,
    ) -> None:
        self.items = items
        self.root = Path(root)
        self.crop = crop
        self.train = train
        self.use_color = use_color
        self.label_shift = label_shift

    def __len__(self) -> int:
        return len(self.items)

    def _paths(self, rel: str) -> tuple[Path, Path]:
        img = self.root / rel
        msk = self.root / rel.replace("images_png", "masks_png")
        return img, msk

    def _augment(self, img: np.ndarray, msk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = msk.shape
        if self.crop and (h > self.crop or w > self.crop):
            y = random.randint(0, h - self.crop)
            x = random.randint(0, w - self.crop)
            img = img[y : y + self.crop, x : x + self.crop]
            msk = msk[y : y + self.crop, x : x + self.crop]

        if random.random() < 0.5:
            img, msk = img[:, ::-1], msk[:, ::-1]
        if random.random() < 0.5:
            img, msk = img[::-1], msk[::-1]
        k = random.randint(0, 3)
        if k:
            img, msk = np.rot90(img, k), np.rot90(msk, k)
        return np.ascontiguousarray(img), np.ascontiguousarray(msk)

    def _color(self, img: np.ndarray) -> np.ndarray:
        # 亮度/对比度/饱和度/色相抖动，幅度按 doc 4.2
        f = np.float32(img)
        f *= 1.0 + random.uniform(-0.2, 0.2)
        mean = f.mean(axis=(0, 1), keepdims=True)
        f = (f - mean) * (1.0 + random.uniform(-0.2, 0.2)) + mean
        gray = f.mean(axis=2, keepdims=True)
        f = gray + (f - gray) * (1.0 + random.uniform(-0.2, 0.2))
        f += random.uniform(-0.05, 0.05) * 255.0
        return np.clip(f, 0, 255).astype(np.uint8)

    def __getitem__(self, i: int) -> dict:
        rel = self.items[i]
        img_path, msk_path = self._paths(rel)
        with Image.open(img_path) as im:
            img = np.asarray(im.convert("RGB"), dtype=np.uint8)
        with Image.open(msk_path) as mk:
            raw = np.asarray(mk.convert("L"), dtype=np.uint8)

        msk = remap_mask(raw) if self.label_shift else raw
        if self.train:
            img, msk = self._augment(img, msk)
            if self.use_color:
                img = self._color(img)

        x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x.transpose(2, 0, 1)).float()
        y = torch.from_numpy(msk).long()
        return {"image": x, "mask": y, "name": rel}


def quick_check() -> None:
    """自检：抽 3 张，打印形状、mask 取值、忽略像素比例。"""
    imgs = sorted((RAW / "Train" / "Urban" / "images_png").glob("*.png"))
    rels = [str(p.relative_to(RAW)).replace("\\", "/") for p in imgs[:3]]
    ds = LoveDADataset(rels, crop=512, train=False)
    for i in range(len(ds)):
        s = ds[i]
        y = s["mask"]
        vals = torch.unique(y).tolist()
        ignore = float((y == IGNORE_INDEX).float().mean())
        print(
            f"{s['name']}: image {tuple(s['image'].shape)} dtype={s['image'].dtype} "
            f"mask {tuple(y.shape)} 取值={vals} ignore={ignore:.2%}"
        )


if __name__ == "__main__":
    quick_check()
