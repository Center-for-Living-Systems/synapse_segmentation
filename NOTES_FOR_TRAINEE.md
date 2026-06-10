# Notes for the Trainee

This document is a practical guide for anyone picking up this project.
It explains what the code does, why certain decisions were made, what the
current limitations are, and what to try next.

---

## What this project does

We train a UNet to do **pixel-wise segmentation** of synapse tissue microscopy images.
There are three classes:

| Class | Label | Description |
|---|---|---|
| Background | 0 | Everything that is not a labelled cell |
| G cells | 1 | A specific cell type, very rare |
| P cells | 2 | Another cell type, more common |

Input images are **16-bit grayscale TIF** files taken on a fluorescence microscope.
Annotations are **ImageJ polygon ROIs** stored as zip files.

The end product of a training run is a `best_model.pth` checkpoint you can use
to segment new images of the same tissue type.

---

## Repository layout

```
synapse_cls/
├── data/
│   └── dataset.py          ← patch extraction, augmentation, dataset class
├── models/
│   ├── unet.py             ← from-scratch UNet architecture
│   ├── builder.py          ← factory: build_model(), get_num_classes()
│   └── losses.py           ← DiceLoss, ComboLoss
├── utils/
│   ├── roi_utils.py        ← reads ImageJ ROI zips → numpy masks
│   └── metrics.py          ← class weights, IoU, loss/IoU curve plots
├── train.py                ← training script (calls the library above)
├── visualize.py            ← diagnostic figures (patches, predictions)
├── check_overlap.py        ← spatial patch overlap analysis
├── check_similarity.py     ← pixel-intensity patch similarity analysis
├── configs/                ← one YAML per experiment run
├── outputs/                ← all experiment outputs (not committed)
└── logs/                   ← training stdout logs
```

`train.py` and `visualize.py` are **scripts** (entry points you run from the
command line). The actual reusable logic lives in `data/`, `models/`, and `utils/`.

---

## How to run an experiment

```bash
# 1. activate the environment
conda activate subcellae-cuda

# 2. train
python train.py --config configs/run02_smp_resnet34_3cls_ce_random.yaml

# 3. visualise results
python visualize.py --config configs/run02_smp_resnet34_3cls_ce_random.yaml \
                    --checkpoint outputs/run02_smp_resnet34_3cls_ce_random/best_model.pth
```

To add a new run:

1. Copy the closest config from `configs/` and rename it `run{NN}_<descriptor>.yaml`.
2. Edit the hyperparameters and set `output_dir: ./outputs/run{NN}_<descriptor>`.
3. Run `python train.py --config configs/run{NN}_<descriptor>.yaml | tee logs/run{NN}_<descriptor>.log`.

---

## What each config key does

```yaml
data_dir: /path/to/data      # folder containing *F.tif and *s.zip files
patch_size: 256              # crop size in pixels (use 224 for smp encoders)
patches_per_image: 100       # number of patches sampled per TIF at startup
val_image_idx: 3             # 0-based index of the held-out validation image
seed: 42

model: smp_unet              # "smp_unet" or "custom_unet"
encoder_name: resnet34       # smp_unet only — any timm/torchvision encoder
encoder_weights: imagenet    # smp_unet only — pretrained weights
base_features: 64            # custom_unet only — channel width at level 1

batch_size: 8
num_workers: 4
lr: 1.0e-4
weight_decay: 1.0e-4
epochs: 100
save_every: 10               # periodic checkpoint interval

binary: false                # true = merge G+P → "Cell" (Background vs Cell)
loss: combo                  # "ce" or "combo"
dice_weight: 0.5             # Dice fraction in combo loss
annotated_fraction: 0.7      # fraction of patches guaranteed to contain cells
aug_elastic: true            # elastic deformation (from the original UNet paper)
aug_intensity: true          # brightness/contrast jitter

output_dir: ./outputs/run{NN}_<descriptor>
```

---

## Current results (12 runs)

All 3-class runs use val image = AnnotatedSyrinx4F.

| Run | Model | Task | Loss | Sampling | Best ep | Val loss | G IoU | P IoU | Fg IoU |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Custom UNet (256px) | 3-class | CE | Random | 41 | 1.228 | 0.053 | 0.745 | 0.399 |
| 2 | ResNet34 ImageNet (224px) | 3-class | CE | Random | 21 | 0.821 | 0.179 | 0.745 | 0.462 |
| 3 | ResNet34 ImageNet (224px) | Binary | CE | Random | 50 | 0.139 | — | 0.764 | 0.764 |
| 4 | ResNet34 ImageNet (224px) | Binary | Combo | Ann-aware | 34 | 0.244 | — | 0.780 | **0.780** |
| 5 | ResNet34 ImageNet (224px) | 3-class | Combo | Ann-aware | 30 | 1.045 | 0.136 | 0.757 | 0.447 |
| 6 | Custom UNet (256px) | 3-class | Combo | Ann-aware | 46 | 0.714 | 0.155 | 0.741 | 0.448 |
| 7 | EfficientNet-B0 ImageNet | 3-class | CE | Random | 19 | 0.845 | 0.137 | 0.752 | 0.445 |
| 8 | MobileNetV2 ImageNet | 3-class | CE | Random | 47 | 1.184 | 0.107 | 0.749 | 0.428 |
| 9 | EfficientNet-B0 ImageNet | 3-class | CE | Ann-aware | 14 | 0.890 | 0.083 | 0.732 | 0.407 |
| 10 | MobileNetV2 ImageNet | 3-class | CE | Ann-aware | 23 | 1.164 | 0.031 | 0.702 | 0.366 |
| 11 | EfficientNet-B0 ImageNet | 3-class | Combo | Ann-aware | 13 | 0.621 | 0.144 | 0.747 | 0.446 |
| 12 | MobileNetV2 ImageNet | 3-class | Combo | Ann-aware | 15 | 0.936 | 0.033 | 0.725 | 0.379 |

"Fg IoU" = mean IoU over foreground classes only (excludes Background).

---

## Known limitations and why

### 1. Only 4 images — the fundamental bottleneck

We have 4 images total, one of which is held out for validation.
That leaves **3 training images**. All architectures and all loss combinations
plateau at fg IoU ~0.33–0.46 for the 3-class task. This is not a model
problem — it is a data problem. More annotated images are the highest-leverage
next step.

### 2. G-cell IoU is very low (0.03–0.18)

G cells are extremely rare: they occupy roughly 1/170 of all pixels in the
training set. Class weights help, but with so few positive examples the model
cannot learn a reliable G-cell detector. In binary mode (runs 3-4), where G
and P are merged, performance jumps to ~0.78 — the signal is there, just not
enough to separate the two cell types.

### 3. Patch overlap on small images

Images 2 and 4 are 512×512 pixels. A 256×256 patch covers 1/4 of the image,
so 100 randomly sampled patches from a 512×512 image are almost all copies of
one of four possible tiles (median pairwise spatial overlap ~42%). This is
unavoidable until more or larger images are available.

### 4. Train/val split is image-level (by design)

Patches are split by **whole image**, not randomly across all patches. This
prevents spatial leakage: because adjacent patches on the same image share
tissue structure, a random patch split would let the model "memorise" texture
from nearby patches. Image-level split is the correct approach even though it
means evaluating on a single image.

### 5. Annotation-aware sampling + CE alone hurts

Runs 9 and 10 (ann-aware + CE) performed *worse* than random sampling.
Annotation-aware sampling pushes cell patches to 70% of the batch, but CE loss
responds by biasing towards background again. Combo loss (with Dice) counteracts
this — runs 11-12 recover performance. If you use annotation-aware sampling,
use combo loss too.

---

## What to try next

In rough priority order:

### More annotated images (highest leverage)
Everything else is secondary. Even 10 images would transform the G-cell problem
from impossible to tractable.

### Binary-then-classify two-stage approach
Train a binary model (Background vs Cell, which already works well at ~0.78 IoU)
and a separate G-vs-P classifier on the detected cell regions. This decouples
the hard segmentation problem from the harder identity problem and avoids the
extreme G-class imbalance.

### Test-time augmentation (TTA)
At inference, run the model on several augmented versions of each patch (flips,
rotations) and average the softmax outputs. This is free performance at the cost
of inference speed.

### Semi-supervised / self-training
If you have unannotated images, train on labelled data, generate pseudo-labels
on unannotated images (thresholding by model confidence), and fine-tune with
the expanded set.

### Full-image inference with sliding window
`visualize.py` predicts on the same fixed patches used during training. For
deployment you want a sliding-window inference that tiles the full image with
overlap and averages the overlapping predictions. This avoids boundary artefacts.

### Post-processing
Connected-component filtering (remove predictions smaller than a minimum cell
area) can eliminate false positives without retraining.

### Focal loss for the G class
Focal loss (`(1-p)^γ × CE`) down-weights easy background examples automatically.
It is an alternative to hand-tuned inverse-frequency weights and can be more
stable when class frequencies change across images.

---

## Where to look for help

- `data/dataset.py` — start here to understand how patches are created and
  how augmentation works.
- `models/losses.py` — read the docstrings to understand why Dice is computed
  over foreground only and why that matters.
- `utils/roi_utils.py` — explains how ImageJ ROI zips are converted to numpy
  masks; useful if you have new annotation formats.
- `README.md` — model architecture diagram, training settings, config reference.
