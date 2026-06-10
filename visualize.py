"""
Visualize dataset and model predictions.

Outputs saved to <output_dir>/vis/:
  01_roi_masks.png          full-resolution image + mask overlay for every TIF
  02_train_patches.png      all training patches (image | mask)
  03_val_patches.png        all validation patches (image | mask)
  04_predictions_train.png  model predictions on training patches (image | gt | pred)
  05_predictions_val.png    model predictions on validation patches (image | gt | pred)

Sections 04/05 require --checkpoint (or outputs/best_model.pth).

Usage:
    python visualize.py --config configs/run02_smp_resnet34_3cls_ce_random.yaml
    python visualize.py --config configs/run02_smp_resnet34_3cls_ce_random.yaml --checkpoint outputs/run02_smp_resnet34_3cls_ce_random/best_model.pth
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.dataset import CLASS_NAMES, NUM_CLASSES, PatchDataset, load_all_images
from models.builder import build_model

# ── colors ────────────────────────────────────────────────────────────────────
# float RGB: 0=Background (dark), 1=G (green), 2=P (magenta)
_COLORS = np.array([[0.08, 0.08, 0.08], [0.0, 0.82, 0.0], [0.82, 0.0, 0.82]], dtype=np.float32)


def _colorize(mask: np.ndarray) -> np.ndarray:
    return _COLORS[mask]  # (H, W, 3)


def _overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    rgb = np.stack([img, img, img], axis=-1)
    fg = mask > 0
    result = rgb.copy()
    result[fg] = rgb[fg] * (1 - alpha) + _colorize(mask)[fg] * alpha
    return np.clip(result, 0, 1)


def _legend():
    return [
        mpatches.Patch(color=_COLORS[1], label="G"),
        mpatches.Patch(color=_COLORS[2], label="P"),
    ]


# ── section 1 ─────────────────────────────────────────────────────────────────

def save_roi_masks(images: list[dict], out_path: Path) -> None:
    n = len(images)
    fig, axes = plt.subplots(n, 2, figsize=(12, 5 * n), squeeze=False)

    for row, d in enumerate(images):
        axes[row, 0].imshow(d["image"], cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f'{d["name"]}\nOriginal (16-bit normalized)', fontsize=9)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(_overlay(d["image"], d["mask"]))
        axes[row, 1].set_title(f'{d["name"]}\nMask overlay', fontsize=9)
        axes[row, 1].axis("off")
        axes[row, 1].legend(handles=_legend(), loc="lower right", fontsize=8)

    fig.suptitle("ROI masks  ·  Green = G  ·  Magenta = P", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ── sections 2/3 ──────────────────────────────────────────────────────────────

def save_patch_grid(
    patches: list[tuple[np.ndarray, np.ndarray]],
    title: str,
    out_path: Path,
    cols: int = 8,
) -> None:
    """Two sub-rows per patch block: top = raw image, bottom = mask overlay."""
    n = len(patches)
    rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 2.2, rows * 4.2), squeeze=False)
    axes = axes.reshape(rows, 2, cols)

    for idx, (img, mask) in enumerate(patches):
        r, c = divmod(idx, cols)
        axes[r, 0, c].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[r, 0, c].axis("off")
        axes[r, 1, c].imshow(_overlay(img, mask))
        axes[r, 1, c].axis("off")

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, 0, c].axis("off")
        axes[r, 1, c].axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ── sections 4/5 ──────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    patches: list[tuple[np.ndarray, np.ndarray]],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 16,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Returns list of (img, gt_mask, pred_mask) as numpy arrays."""
    ds = PatchDataset(patches, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    results = []
    for imgs, masks in loader:
        preds = model(imgs.to(device)).argmax(1).cpu().numpy()
        imgs_np = imgs.squeeze(1).numpy()
        masks_np = masks.numpy()
        for i in range(len(imgs_np)):
            results.append((imgs_np[i], masks_np[i], preds[i]))
    return results


def save_predictions(
    results: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    title: str,
    out_path: Path,
    cols: int = 5,
) -> None:
    """Three sub-rows per block: image | ground truth overlay | prediction overlay."""
    n = len(results)
    rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(rows * 3, cols, figsize=(cols * 2.5, rows * 6.5), squeeze=False)
    axes = axes.reshape(rows, 3, cols)

    for idx, (img, gt, pred) in enumerate(results):
        r, c = divmod(idx, cols)
        axes[r, 0, c].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[r, 0, c].set_title("image", fontsize=6)
        axes[r, 0, c].axis("off")

        axes[r, 1, c].imshow(_overlay(img, gt))
        axes[r, 1, c].set_title("ground truth", fontsize=6)
        axes[r, 1, c].axis("off")

        axes[r, 2, c].imshow(_overlay(img, pred))
        axes[r, 2, c].set_title("prediction", fontsize=6)
        axes[r, 2, c].axis("off")

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        for p in range(3):
            axes[r, p, c].axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/run02_smp_resnet34_3cls_ce_random.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["output_dir"]) / "vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    val_idx = cfg.get("val_image_idx", 3)

    print("Loading images and extracting patches...")
    images = load_all_images(
        data_dir=cfg["data_dir"],
        patch_size=cfg["patch_size"],
        patches_per_image=cfg["patches_per_image"],
        seed=cfg["seed"],
    )

    train_patches = [p for i, d in enumerate(images) for p in d["patches"] if i != val_idx]
    val_patches   = [p for i, d in enumerate(images) for p in d["patches"] if i == val_idx]
    val_name = images[val_idx]["name"] if val_idx < len(images) else "?"
    print(f"Train patches: {len(train_patches)}  |  Val patches: {len(val_patches)} ({val_name})")

    print("\n[1/5] ROI mask overlays...")
    save_roi_masks(images, out_dir / "01_roi_masks.png")

    print("[2/5] Training patch grid...")
    save_patch_grid(
        train_patches,
        f"Training patches  (n={len(train_patches)}, 3 images)",
        out_dir / "02_train_patches.png",
    )

    print("[3/5] Validation patch grid...")
    save_patch_grid(
        val_patches,
        f"Validation patches  (n={len(val_patches)}, {val_name})",
        out_dir / "03_val_patches.png",
    )

    ckpt_path = Path(args.checkpoint) if args.checkpoint else Path(cfg["output_dir"]) / "best_model.pth"
    if not ckpt_path.exists():
        print(f"\n[4-5/5] No checkpoint at {ckpt_path} — skipping prediction figures.")
        print("        Train the model first, then re-run visualize.py.")
        return

    print(f"\n[4/5] Loading checkpoint: {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # use the config stored in the checkpoint so the architecture always matches
    model = build_model(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Epoch {ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}")

    print("[4/5] Predicting on train patches...")
    train_results = run_inference(train_patches, model, device, cfg["batch_size"])
    save_predictions(
        train_results,
        f"Train predictions  (n={len(train_results)})  epoch={ckpt['epoch']}",
        out_dir / "04_predictions_train.png",
    )

    print("[5/5] Predicting on val patches...")
    val_results = run_inference(val_patches, model, device, cfg["batch_size"])
    save_predictions(
        val_results,
        f"Val predictions  ({val_name}, n={len(val_results)})  epoch={ckpt['epoch']}",
        out_dir / "05_predictions_val.png",
    )

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
