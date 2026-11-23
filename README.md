# 高分辨率影像语义分割与跨域泛化研究

基于 LoveDA 数据集的语义分割全流程实验：UNet / DeepLabV3+ / SegFormer-B2 三模型对比、**跨域泛化控制变量实验**、
错误归因分析与 ONNX 部署测速，另附 RemoteCLIP 零样本与线性探针的对照评估。

> 单卡（RTX 5060 Laptop 8 GB）完成全部训练与评测。仓库只包含代码，**不含数据集与权重**。

---

## 核心结论（一行版）

同一批数据、同样的训练样本量，**只改划分方式**（随机划分 vs 跨城划分），test mIoU 从 0.6520 掉到 0.5331，
**分布偏移代价 11.89 点**；且掉点高度不均匀——林地 −24.9、裸地 −23.8、农田 −18.9，而**建筑只掉 −1.4**。

---

## 关键结果

### 模型对比（test 集 835 张，一次出数、不调参）

| 实验 | mIoU | mIoU（不含背景） | 像素准确率 | Kappa |
|---|---|---|---|---|
| UNet（ResNet34，从零训练） | 0.4287 | 0.4167 | 0.6338 | 0.4954 |
| DeepLabV3+（ImageNet 预训练） | 0.5026 | 0.4978 | 0.6827 | 0.5705 |
| SegFormer-B2（ADE20K 预训练） | 0.5211 | 0.5182 | 0.7068 | 0.6048 |
| **SegFormer-B2 + 颜色增强** | **0.5465** | 0.5426 | 0.7286 | 0.6332 |

**归因**：预训练权重贡献 **+7.4 点**（UNet→DeepLabV3+），骨干结构差异 **+1.9 点**（DeepLabV3+→SegFormer），
颜色增强 **+2.4 点**。在算力受限时，这个排序直接决定资源该花在哪。

### 跨域泛化对照（M3）

两臂**同模型、同 2271 张训练量、同城乡构成、同超参与种子**，训练/测试交集断言为 0，唯一变量是划分方式：

| 实验 | 训练 | 测试 | mIoU |
|---|---|---|---|
| 随机划分 | 2271 张 | 839 张（同分布） | **0.6520** |
| 官方（跨城）划分 | 2271 张 | 1669 张（跨城） | **0.5331** |

按类别拆开的掉点分布：

| 类别 | 随机划分 | 跨城划分 | 掉点 |
|---|---|---|---|
| 林地 | 0.663 | 0.413 | **−24.9** |
| 裸地 | 0.539 | 0.301 | **−23.8** |
| 农田 | 0.739 | 0.550 | −18.9 |
| 背景 | 0.612 | 0.553 | −5.9 |
| 道路 | 0.617 | 0.572 | −4.5 |
| 水体 | 0.751 | 0.714 | −3.8 |
| 建筑 | 0.644 | 0.630 | **−1.4** |

**解释**：屋顶的形状与材质跨城市稳定；植被覆盖度、裸土颜色、物候随气候与季节变化。
颜色增强的增益正好集中在农田（+4.2）/ 背景（+3.8）/ 林地（+3.6）、建筑仅 +0.2，方向一致。

### 多模态对照（RemoteCLIP，224 小块口径，macro-F1）

| 实验 | OA | macro-F1 | balanced acc |
|---|---|---|---|
| RemoteCLIP 零样本 | 0.6296 | 0.5690 | 0.6403 |
| 线性探针（冻结编码器） | 0.7775 | **0.7792** | 0.7754 |

5 种 prompt 模板消融波动仅 3.5 点 → 瓶颈不在措辞而在图文对齐。
零样本在“其他地表”兜底类上召回为 **0**，原因是该类无视觉原型；换成线性分类器后立刻到 0.63。

### 部署（512 输入）

| 实现 | 延迟 |
|---|---|
| PyTorch + CUDA | 21.9 ms |
| ONNXRuntime CUDA | **21.4 ms** |
| ONNXRuntime CPU | 593 ms |

ONNX 导出最大数值误差 6.2e−4。

---

## 三个最值得看的设计

1. **标签体系纠错**：数据集文档称标签为 0–6，实际取值为 0–7（0 为 no-data）。
   核对后统一做 `raw−1` 映射并将 0 置为 `ignore_index`，避免整条流水线类别错位——这是会导致所有指标作废的高危缺陷。
2. **控制变量的可验证性**：M3 两臂的“只有一处不同”不是靠叙述保证的，而是划分脚本里带交集断言、
   样本量与域构成枚举比对、超参与种子共用同一份 yaml。
3. **选模与报数的纪律**：dev（834 张）只用于选模型，test（835 张）一次出数不再调参。
   两者差 0.1–0.6 点，说明在 dev 上选模型没有过拟合。

---

## 目录结构

```
src/
  loveda.py          类别常量、原始标签映射
  download.py        数据校验（张数逐格对齐 / md5 全量去重 / 尺寸检查）
  splits.py          划分生成（含训练-测试交集断言）
  splits_m3.py       M3 控制变量划分
  quality.py         类别像素占比、no-data、重复图、跨域色调统计
  eda.py             样例图墙、类别占比、共现矩阵
  dataset.py         Dataset 与数据增强
  models.py          UNet / DeepLabV3+ / SegFormer-B2 统一构建
  train.py           训练循环（AMP、梯度累积、调度、断点续训、早停）
  evaluate.py        滑窗推理与指标计算（mIoU / Kappa / 混淆矩阵）
  error_analysis.py  边界距离分桶、面积分桶召回、Top-5 混淆对
  noise_review.py    最大错误块人工复核表
  clip_eval.py       RemoteCLIP 零样本 / 线性探针 / prompt 消融 / 检索
  export_onnx.py     ONNX 导出与 ORT 测速
configs/             各实验的超参配置
```

## 快速开始

```bash
pip install -r requirements.txt

# 1) 数据校验（需先把 LoveDA 放到 data/raw/LoveDA）
python -m src.download
python -m src.splits
python -m src.quality

# 2) 训练
python -m src.train --config configs/segformer_b2.yaml

# 3) 评测
python -m src.evaluate --config configs/segformer_b2.yaml --split test

# 4) 错误分析 / 部署
python -m src.error_analysis
python -m src.export_onnx --model segformer_b2 --ckpt <path>
```

## 数据

[LoveDA](https://github.com/Junjue-Wang/LoveDA)（NeurIPS 2021 Datasets & Benchmarks），
5987 张 1024×1024、0.3 m 分辨率，南京 / 武汉 / 常州，7 类地物。
许可证 **CC BY-NC-SA 4.0**，仅用于学习研究；本仓库不包含数据与权重。

## 已知局限

- 单卡 8 GB 限制 batch size，SegFormer 用 `batch 2 × accum 4` 等效 batch 8，未做多卡验证。
- 官方 Test 集不提供标注，因此“测试集”实为 Val 按城乡分层切出的 835 张。
- 颜色增强只能覆盖外观偏移的一个维度，跨域问题未真正解决（这是后续域适应工作的起点）。
- 未做 INT8 量化与轻量骨干替换，CPU 推理仍是瓶颈（593 ms）。
