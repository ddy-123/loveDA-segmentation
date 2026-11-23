# LoveDA Semantic Segmentation & Cross-Domain Generalization

An end-to-end semantic segmentation study on the [LoveDA](https://github.com/Junjue-Wang/LoveDA) dataset:
a three-model comparison (UNet / DeepLabV3+ / SegFormer-B2), a **controlled cross-domain generalization experiment**,
bucket-wise error analysis, and ONNX deployment benchmarking. A RemoteCLIP zero-shot vs. linear-probe
comparison is included as a multimodal baseline.

> Everything was trained and evaluated on a **single 8 GB laptop GPU**. This repository contains code only —
> no datasets, no checkpoints.

---

## Headline result

Holding the model, the training set size (2,271 tiles), the urban/rural composition, the hyperparameters and the
random seed constant, and changing **only the train/test split**, test mIoU drops from **0.6520 to 0.5331** —
a **distribution-shift cost of 11.89 points**.

The damage is highly uneven: forest **−24.9**, barren **−23.8**, agriculture **−18.9**, but building only **−1.4**.
Roof geometry and material transfer across cities; vegetation cover, bare-soil colour and phenology do not.

---

## Results

### Model comparison (835 held-out tiles, single evaluation pass, no tuning)

| Experiment | mIoU | mIoU (no background) | Pixel acc. | Kappa |
|---|---|---|---|---|
| UNet (ResNet34, from scratch) | 0.4287 | 0.4167 | 0.6338 | 0.4954 |
| DeepLabV3+ (ImageNet pre-trained) | 0.5026 | 0.4978 | 0.6827 | 0.5705 |
| SegFormer-B2 (ADE20K pre-trained) | 0.5211 | 0.5182 | 0.7068 | 0.6048 |
| **SegFormer-B2 + colour augmentation** | **0.5465** | 0.5426 | 0.7286 | 0.6332 |

**Attribution:** pre-training contributes **+7.4 points**, backbone architecture only **+1.9**, colour augmentation
**+2.4**. Under a fixed compute budget this ordering dictates where the effort should go.

### Cross-domain control experiment

| Arm | Train | Test | mIoU |
|---|---|---|---|
| Random split | 2,271 tiles | 839 tiles (i.i.d.) | **0.6520** |
| Official (cross-city) split | 2,271 tiles | 1,669 tiles (cross-city) | **0.5331** |

Per-class degradation:

| Class | Random split | Cross-city | Δ |
|---|---|---|---|
| Forest | 0.663 | 0.413 | **−24.9** |
| Barren | 0.539 | 0.301 | **−23.8** |
| Agriculture | 0.739 | 0.550 | −18.9 |
| Background | 0.612 | 0.553 | −5.9 |
| Road | 0.617 | 0.572 | −4.5 |
| Water | 0.751 | 0.714 | −3.8 |
| Building | 0.644 | 0.630 | **−1.4** |

Supporting evidence: colour augmentation gains concentrate on agriculture (+4.2), background (+3.8) and
forest (+3.6) while building gains only +0.2 — consistent with the hypothesised appearance shift.

### Multimodal baseline (RemoteCLIP, 224×224 patches, macro-F1)

| Experiment | OA | macro-F1 | Balanced acc. |
|---|---|---|---|
| RemoteCLIP zero-shot | 0.6296 | 0.5690 | 0.6403 |
| Linear probe (frozen encoder) | 0.7775 | **0.7792** | 0.7754 |

Five prompt templates vary by only 3.5 macro-F1 points, so the bottleneck is alignment rather than wording.
Zero-shot recall on the "other surface" catch-all class is **0** — that class has no visual prototype; a linear
classifier recovers it to 0.63, showing the image features do carry the information.

### Deployment (512×512 input)

| Implementation | Latency |
|---|---|
| PyTorch + CUDA | 21.9 ms |
| ONNXRuntime CUDA | **21.4 ms** |
| ONNXRuntime CPU | 593 ms |

Maximum ONNX numerical deviation: 6.2e−4.

---

## Three design decisions worth a look

1. **Label-scheme audit.** The dataset documentation states labels 0–6; the masks actually contain 0–7, where 0
   is no-data. Every mask is remapped as `raw − 1` and 0 becomes `ignore_index`. Skipping this silently shifts
   every class by one and invalidates the entire pipeline.
2. **Verifiable controls, not narrative.** The two arms differ in exactly one factor — enforced by an
   intersection assertion in the split script, enumerated sample-count and domain-composition checks, and a
   shared YAML for hyperparameters and seed.
3. **Discipline in model selection.** `dev` (834 tiles) is used only to pick a model; `test` (835 tiles) is
   evaluated once, with no tuning. The gap between them is 0.1–0.6 points, indicating no selection overfitting.

---

## Repository layout

```
src/
  loveda.py          class constants and raw-label mapping
  download.py        data audit (per-cell alignment, full md5 dedupe, size checks)
  splits.py          split generation, with train/test intersection assertion
  splits_m3.py       splits for the controlled experiment
  quality.py         class pixel ratios, no-data statistics, duplicate detection, cross-city colour stats
  eda.py             sample grids, class distribution, co-occurrence matrices
  dataset.py         Dataset and augmentation pipeline
  models.py          unified builders for UNet / DeepLabV3+ / SegFormer-B2
  train.py           training loop (AMP, gradient accumulation, schedulers, resume, early stopping)
  evaluate.py        sliding-window inference and metrics (mIoU / Kappa / confusion matrix)
  error_analysis.py  boundary-distance bins, area-binned recall, top-5 confusion pairs
  noise_review.py    human review sheets for the worst error blobs
  clip_eval.py       RemoteCLIP zero-shot / linear probe / prompt ablation / retrieval
  export_onnx.py     ONNX export and ONNXRuntime benchmarking
configs/             per-experiment hyperparameter files
```

## Quick start

```bash
pip install -r requirements.txt

# 1) data audit (place LoveDA under data/raw/LoveDA first)
python -m src.download
python -m src.splits
python -m src.quality

# 2) train
python -m src.train --config configs/segformer_b2.yaml

# 3) evaluate
python -m src.evaluate --config configs/segformer_b2.yaml --split test

# 4) error analysis / export
python -m src.error_analysis
python -m src.export_onnx --model segformer_b2 --ckpt <path>
```

## Data

[LoveDA](https://github.com/Junjue-Wang/LoveDA) (NeurIPS 2021 Datasets & Benchmarks): 5,987 tiles at
1024×1024, 0.3 m resolution, covering Nanjing, Wuhan and Changzhou, 7 land-cover classes.
Licensed **CC BY-NC-SA 4.0**, research use only. No data or weights are redistributed here.

## Known limitations

- An 8 GB card forces small batches; SegFormer runs at `batch 2 × accum 4` (effective batch 8). No multi-GPU
  validation was performed.
- The official test split ships images without annotations, so the "test set" here is 835 tiles carved from the
  validation split, stratified by urban/rural.
- Colour augmentation covers only one axis of the appearance shift. Domain adaptation is the natural next step
  and is not implemented.
- No INT8 quantisation and no lightweight backbone swap; CPU inference at 593 ms remains the bottleneck.
