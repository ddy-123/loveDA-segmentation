"""三个模型的统一构建入口。

B1 UNet(ResNet34, 从零) / B2 DeepLabV3+(ResNet50, ImageNet) / M1 SegFormer-B2(ADE20K)。
权重来源：HuggingFace 主站不可达，统一走 hf-mirror.com（见 README“数据与环境”）。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# 必须在 import transformers / smp 之前设置。
# 用户目录下的 .cache 在本机不可写（WinError 5），所以缓存一律放项目内，
# 这样任何入口（train/evaluate/error_analysis/export_onnx）都不会再踩同一个坑。
_PROJ = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HOME", str(_PROJ / ".cache" / "hf"))
os.environ.setdefault("TORCH_HOME", str(_PROJ / ".cache" / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(_PROJ / ".cache" / "mpl"))
for _d in (os.environ["HF_HOME"], os.environ["TORCH_HOME"], os.environ["MPLCONFIGDIR"]):
    Path(_d).mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 7

# 各模型的预训练权重（同时用来确定"是 B2 还是 B0"这种结构差异）
PRETRAINED = {
    "segformer_b2": "nvidia/segformer-b2-finetuned-ade-512-512",
}


class SegFormerWrapper(nn.Module):
    """transformers 的 SegFormer 输出 logits 是 1/4 分辨率，这里上采样回输入尺寸。"""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: str | None = None,
        arch: str | None = None,
    ) -> None:
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        if pretrained:
            self.net = SegformerForSemanticSegmentation.from_pretrained(
                pretrained,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            # 关键：不加载权重时也必须用正确的结构（B2 而不是默认的 B0），
            # 否则评估时 state_dict 会因隐层维度不同而全部 size mismatch。
            ref = arch or PRETRAINED["segformer_b2"]
            cfg = SegformerConfig.from_pretrained(ref)
            cfg.num_labels = num_classes
            cfg.id2label = {i: f"class{i}" for i in range(num_classes)}
            cfg.label2id = {v: k for k, v in cfg.id2label.items()}
            self.net = SegformerForSemanticSegmentation(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(pixel_values=x).logits
        return nn.functional.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_model(
    name: str,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    arch: str | None = None,
) -> nn.Module:
    name = name.lower()
    if name in {"segformer_b2", "segformer"}:
        ckpt = (arch or PRETRAINED["segformer_b2"]) if pretrained else None
        return SegFormerWrapper(num_classes, ckpt, arch=arch)

    import segmentation_models_pytorch as smp

    if name in {"deeplabv3plus", "deeplab"}:
        return smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet" if pretrained else None,
            classes=num_classes,
        )
    if name in {"unet", "b1"}:
        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,  # B1 定义就是从零
            classes=num_classes,
        )
    raise ValueError(f"unknown model: {name}")


def count_params(model: nn.Module) -> dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_M": round(total / 1e6, 2), "trainable_M": round(trainable / 1e6, 2)}
