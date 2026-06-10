# Synapse G/P Segmentation — UNet

Pixel-wise segmentation and classification of two cell types (G and P) in 16-bit fluorescence microscopy images of synapse tissue. Three-class output: **Background**, **G**, **P**.

---

## Data

Images and annotations live in a separate data directory (not committed to this repo).

| File pattern | Description |
|---|---|
| `AnnotatedSyrinxNF.tif` | 16-bit grayscale TIF, single frame |
| `AnnotatedSyrinxNGs.zip` | ImageJ ROI zip — G-cell polygon annotations |
| `AnnotatedSyrinxNPs.zip` | ImageJ ROI zip — P-cell polygon annotations |

**Current dataset (4 images):**

| Image | Size | G | P | Split |
|---|---|---|---|---|
| AnnotatedSyrinx1F | 1386×2154 | — | ✓ | Train |
| AnnotatedSyrinx2F | 512×512 | ✓ | ✓ | Train |
| AnnotatedSyrinx3F | 1536×1536 | ✓ | ✓ | Train |
| AnnotatedSyrinx4F | 512×512 | ✓ | ✓ | **Val** |

Annotations are ImageJ polygon ROIs. Each ROI zip may contain multiple ROI files, one per annotated region. Masks are rendered by filling each polygon: **0 = Background**, **1 = G**, **2 = P**.

---

## Repository Structure

```
synapse_cls/
├── data/
│   └── dataset.py              # patch extraction, augmentation, PatchDataset, train/val split
├── models/
│   ├── unet.py                 # from-scratch UNet architecture
│   ├── builder.py              # build_model(), get_num_classes(), get_class_names()
│   └── losses.py               # DiceLoss, ComboLoss
├── utils/
│   ├── roi_utils.py            # ImageJ ROI zip → numpy mask
│   └── metrics.py              # compute_class_weights(), iou_per_class(), plot_curves()
├── train.py                    # training entry-point script
├── visualize.py                # diagnostic figures (ROI masks, patches, predictions)
├── check_overlap.py            # spatial patch overlap analysis
├── check_similarity.py         # pixel-intensity patch similarity analysis
├── make_ppt.py                 # generate PowerPoint summary presentation
├── configs/                    # one YAML config per experiment run
│   ├── run01_customunet_3cls_ce_random_256px.yaml
│   ├── run02_smp_resnet34_3cls_ce_random.yaml
│   └── ...                     # run03 – run12
├── outputs/                    # all experiment outputs (not committed)
│   ├── run01_.../              # per-run subfolder: checkpoints, metrics.csv, curves, vis/
│   ├── run02_.../
│   └── vis_randompatch_overlapping/   # kept for reference: shows patch overlap problem
├── logs/                       # training stdout logs, one per run
└── environment.yml             # conda environment
```

---

## Model Architecture — UNet

The model is a **4-level symmetric UNet** ([Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)) with batch normalisation, adapted for single-channel 16-bit microscopy input and 3-class pixel-wise output.

### Overall structure

```
Input  (1 × H × W)
  │
  ▼  Encoder (contracting path)
  ├─ DoubleConv  →  64 ch   ─────────────────────────────────── skip1
  ├─ MaxPool2d(2) + DoubleConv  →  128 ch  ──────────────────── skip2
  ├─ MaxPool2d(2) + DoubleConv  →  256 ch  ──────────────────── skip3
  ├─ MaxPool2d(2) + DoubleConv  →  512 ch  ──────────────────── skip4
  └─ MaxPool2d(2) + DoubleConv  →  1024 ch   (bottleneck)
  │
  ▼  Decoder (expanding path)
  ├─ ConvTranspose2d(1024→512) + cat(skip4) + DoubleConv  → 512 ch
  ├─ ConvTranspose2d(512→256)  + cat(skip3) + DoubleConv  → 256 ch
  ├─ ConvTranspose2d(256→128)  + cat(skip2) + DoubleConv  → 128 ch
  └─ ConvTranspose2d(128→64)   + cat(skip1) + DoubleConv  →  64 ch
  │
  └─ Conv2d(64 → num_classes=3, kernel=1)
Output  (3 × H × W)  — class logits
```

### DoubleConv block

Each `DoubleConv` applies two consecutive:
```
Conv2d(3×3, padding=1, bias=False) → BatchNorm2d → ReLU
```

Using `bias=False` before `BatchNorm` is standard practice (the normalisation layer has its own learnable shift).

### Skip connections

At each encoder level the feature map is stored *before* pooling and concatenated with the up-sampled decoder feature map. This preserves fine spatial detail that would otherwise be lost through repeated downsampling — critical for delineating small cell boundaries.

### Upsampling

Transposed convolutions (`ConvTranspose2d`, stride=2) are used instead of bilinear upsampling to keep the upsampling weights learnable.

### Output

A final `Conv2d(1×1)` maps from 64 feature channels to 3 class logits. No softmax is applied inside the model; `CrossEntropyLoss` in PyTorch expects raw logits.

### Parameter count (base_features=64)

| Component | Parameters |
|---|---|
| Encoder (4 levels) | ~9.9 M |
| Bottleneck | ~4.7 M |
| Decoder (4 levels) | ~14.3 M |
| Output conv | ~195 |
| **Total** | **~31 M** |

---

## Training

### Patch-based strategy

Images vary in size (512×512 to 2154×1386). Rather than resizing, fixed-size **256×256 patches** are randomly sampled from each image at dataset construction time. 100 patches per image are sampled using a fixed random seed for reproducibility.

**Note on small images:** 512×512 images only support 4 non-overlapping 256×256 crops. Sampling 100 patches from these images produces heavily redundant content (~42% median pairwise spatial overlap). This will improve with more images.

### Class imbalance

G and P cells occupy a small fraction of each image. Class weights are computed as inverse-frequency over the training set and passed to `CrossEntropyLoss`:

```
weight_c = total_pixels / (num_classes × count_c)
```

Typical weights: Background ≈ 0.36, P ≈ 4.6, G ≈ 61 — reflecting how rare G cells are.

### Augmentation (train only)

- Random horizontal flip
- Random vertical flip
- Random 90° rotation (0°, 90°, 180°, 270°)

### Optimiser and schedule

| Setting | Value |
|---|---|
| Optimiser | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Schedule | Cosine annealing to lr × 0.01 |
| Epochs | 100 |
| Batch size | 8 |

### Train / Val split

**Image-level split** — all patches from one full image are held out for validation (`val_image_idx` in `config.yaml`). This avoids spatial leakage between train and val that would occur with a random patch-level split.

---

## Binary Segmentation Mode

A simpler binary task — Background vs Cell (G and P merged) — is available via `config_binary.yaml`. This avoids the extreme G-class imbalance of the 3-class task and achieves better overall cell detection.

Set `binary: true` in any config to activate. The dataset remaps masks as `0→0, 1→1, 2→1` at `__getitem__` time. Everything else (model, loss, metrics) adapts automatically.

---

## Improvements Implemented

### Combo Loss (CE + Dice)
Standard `CrossEntropyLoss` does not directly optimise IoU, and its minimum occurs early when the model predicts mostly background — making it a poor checkpoint selection criterion. `ComboLoss` combines weighted CE with Dice loss:

```
L = (1 − w) × CE + w × (1 − Dice_fg)
```

Dice is computed only over foreground classes (excludes background) so it focuses gradient signal on the rare cell classes. Default `dice_weight: 0.5`. Checkpoint is now saved by **best foreground IoU** rather than best val loss.

### Annotation-aware Patch Sampling
With random sampling, most patches from large images are pure background (no annotated cells), wasting training budget. `annotated_fraction: 0.7` guarantees 70% of patches contain at least one annotated pixel, sampled via a 2D integral image for O(1) position lookup. The remaining 30% are randomly sampled to maintain background context.

### Elastic Deformation + Intensity Jitter
Applied as numpy transforms in `__getitem__` before tensor conversion:
- **Elastic deformation** (α=80, σ=10): applied 50% of the time, identical warp to image and mask. Based on the augmentation strategy from the original UNet paper.
- **Intensity jitter** (brightness ±25%, contrast ±25%): applied every sample, image only. Simulates acquisition variability across microscopes and sessions.

---

## Usage

### Setup

```bash
conda env create -f environment.yml
conda activate subcellae-cuda
```

### Train

```bash
python train.py --config configs/run02_smp_resnet34_3cls_ce_random.yaml
```

Produces inside the run's `output_dir`:
- `best_model.pth` — best foreground IoU checkpoint (use for inference)
- `best_loss_model.pth` — best val loss checkpoint (useful for diagnosing overfitting)
- `metrics.csv` — per-epoch train/val loss and per-class IoU
- `training_curves.png` — loss and IoU curve figure

### Visualise

```bash
python visualize.py --config configs/run02_smp_resnet34_3cls_ce_random.yaml \
                    --checkpoint outputs/run02_smp_resnet34_3cls_ce_random/best_model.pth
```

Produces figures under `<output_dir>/vis/`:

| File | Content |
|---|---|
| `01_roi_masks.png` | Full-res images with G/P mask overlay |
| `02_train_patches.png` | All training patches (image + mask) |
| `03_val_patches.png` | All val patches |
| `04_predictions_train.png` | Train: image \| ground truth \| prediction |
| `05_predictions_val.png` | Val: image \| ground truth \| prediction |

### Adding a new experiment

1. Copy the closest existing config from `configs/` and rename it `run{NN}_<descriptor>.yaml`
2. Edit the hyperparameters and set `output_dir: ./outputs/run{NN}_<descriptor>`
3. Run `python train.py --config configs/run{NN}_<descriptor>.yaml`
4. Run `python visualize.py --config configs/run{NN}_<descriptor>.yaml`

### Patch overlap analysis

```bash
python check_overlap.py --config configs/run02_smp_resnet34_3cls_ce_random.yaml
```

Checks spatial bounding-box overlap between sampled patches within each image. Useful for diagnosing redundancy when images are small relative to the patch size.

---

## Current Results

All 3-class runs evaluate on AnnotatedSyrinx4F (val image). "Fg IoU" = mean IoU over foreground classes (excludes Background).

| Run | Model | Task | Loss | Sampling | Best ep | Val loss | G IoU | P/Cell IoU | Fg IoU |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Custom UNet (scratch, 256px) | 3-class | CE | Random | 41 | 1.228 | 0.053 | 0.745 | 0.399 |
| 2 | smp ResNet34 ImageNet (224px) | 3-class | CE | Random | 21 | 0.821 | 0.179 | 0.745 | 0.462 |
| 3 | smp ResNet34 ImageNet (224px) | Binary | CE | Random | 50 | 0.139 | — | 0.764 | 0.764 |
| 4 | smp ResNet34 ImageNet (224px) | Binary | Combo | Ann-aware | 34 | 0.244 | — | 0.780 | **0.780** |
| 5 | smp ResNet34 ImageNet (224px) | 3-class | Combo | Ann-aware | 30 | 1.045 | 0.136 | 0.757 | 0.447 |
| 6 | Custom UNet (scratch, 256px) | 3-class | Combo | Ann-aware | 46 | 0.714 | 0.155 | 0.741 | 0.448 |
| 7 | smp EfficientNet-B0 ImageNet | 3-class | CE | Random | 19 | 0.845 | 0.137 | 0.752 | 0.445 |
| 8 | smp MobileNetV2 ImageNet | 3-class | CE | Random | 47 | 1.184 | 0.107 | 0.749 | 0.428 |
| 9 | smp EfficientNet-B0 ImageNet | 3-class | CE | Ann-aware | 14 | 0.890 | 0.083 | 0.732 | 0.407 |
| 10 | smp MobileNetV2 ImageNet | 3-class | CE | Ann-aware | 23 | 1.164 | 0.031 | 0.702 | 0.366 |
| 11 | smp EfficientNet-B0 ImageNet | 3-class | Combo | Ann-aware | 13 | 0.621 | 0.144 | 0.747 | 0.446 |
| 12 | smp MobileNetV2 ImageNet | 3-class | Combo | Ann-aware | 15 | 0.936 | 0.033 | 0.725 | 0.379 |

**Key takeaways:**
- All 3-class runs plateau at fg IoU ~0.37–0.46 regardless of architecture or training recipe. The bottleneck is dataset size (3 training images), not model capacity or loss function.
- Binary segmentation (Cell vs Background) reaches 0.78 IoU — the signal is present, but separating G from P requires more annotated examples.
- G IoU is consistently low (0.03–0.18) due to extreme class imbalance: G cells occupy ~1/170 of all pixels.
- Annotation-aware sampling combined with CE loss (runs 9–10) hurts performance; it requires Combo loss (Dice) to be beneficial (runs 11–12 recover).

---

## Config Reference

```yaml
data_dir: /path/to/data          # folder with *F.tif and *s.zip files
patch_size: 256                  # crop size in pixels
patches_per_image: 100           # random patches sampled per TIF
val_image_idx: 3                 # 0-based index of held-out validation image
seed: 42

base_features: 64                # UNet base channel count
batch_size: 8
num_workers: 4
lr: 1.0e-4
weight_decay: 1.0e-4
epochs: 100
save_every: 10                   # periodic checkpoint interval

binary: false                # true = Background vs Cell (G+P merged)
loss: combo                  # ce | combo
dice_weight: 0.5             # Dice fraction in combo loss
annotated_fraction: 0.7      # fraction of patches guaranteed to contain cells
aug_elastic: true            # elastic deformation augmentation
aug_intensity: true          # random brightness/contrast jitter

output_dir: ./outputs
```
