import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.dataset import NUM_CLASSES


def compute_class_weights(patches, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Inverse-frequency class weights from a list of (image, mask) patch tuples.

    Rare classes (G in particular) receive very high weights to counteract the
    severe class imbalance in the training set.
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, mask in patches:
        for c in range(num_classes):
            counts[c] += int((mask == c).sum())
    weights = counts.sum() / (num_classes * counts + 1e-8)
    return torch.tensor(weights, dtype=torch.float32)


def iou_per_class(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> list[float]:
    """Per-class intersection-over-union. Returns NaN for absent classes."""
    ious = []
    for c in range(num_classes):
        pred_c = preds == c
        tgt_c  = targets == c
        inter  = (pred_c & tgt_c).sum().item()
        union  = (pred_c | tgt_c).sum().item()
        ious.append(inter / union if union > 0 else float("nan"))
    return ious


def plot_curves(csv_path: Path, out_path: Path) -> None:
    """Read a metrics CSV and save a two-panel loss + IoU curve figure."""
    rows = list(csv.DictReader(open(csv_path)))
    epochs     = [int(r["epoch"])        for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    val_loss   = [float(r["val_loss"])   for r in rows]

    iou_cols = [k for k in rows[0] if k.startswith("iou_") and k != "iou_Background"]
    iou_data = {col: [float(r[col]) for r in rows] for col in iou_cols}

    ep = np.array(epochs)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(ep, train_loss, label="train loss", color="steelblue")
    ax1.plot(ep, val_loss,   label="val loss",   color="tomato")
    best_ep = ep[np.nanargmin(val_loss)]
    best_vl = np.nanmin(val_loss)
    ax1.axvline(best_ep, color="tomato", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.text(best_ep + 0.5, best_vl, f" best={best_vl:.4f}\n ep={best_ep}",
             fontsize=7, color="tomato", va="bottom")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train / Val Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    colors = ["seagreen", "mediumpurple", "steelblue", "tomato"]
    for (col, vals), color in zip(iou_data.items(), colors):
        ax2.plot(ep, vals, label=col.replace("iou_", "") + " IoU", color=color)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("IoU")
    ax2.set_ylim(0, 1)
    ax2.set_title("Val IoU — foreground classes")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Training curves", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")
