"""Training entry-point.

Usage:
    python train.py --config configs/run02_smp_resnet34_3cls_ce_random.yaml

Outputs are written to the directory specified by output_dir in the config:
    best_model.pth        checkpoint with best foreground IoU (use for inference)
    best_loss_model.pth   checkpoint with best validation loss
    checkpoint_epochNNNN.pth  periodic snapshots (every save_every epochs)
    metrics.csv           per-epoch train loss, val loss, per-class IoU
    training_curves.png   loss and IoU curves
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from data.dataset import make_train_val_datasets
from models.builder import build_model, get_class_names, get_num_classes
from models.losses import ComboLoss
from utils.metrics import compute_class_weights, iou_per_class, plot_curves


def train(cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, val_ds = make_train_val_datasets(
        data_dir=cfg["data_dir"],
        val_image_idx=cfg.get("val_image_idx", 3),
        patch_size=cfg["patch_size"],
        patches_per_image=cfg["patches_per_image"],
        seed=cfg["seed"],
        binary=cfg.get("binary", False),
        annotated_fraction=cfg.get("annotated_fraction", 0.0),
        aug_elastic=cfg.get("aug_elastic", True),
        aug_intensity=cfg.get("aug_intensity", True),
    )
    print(f"Train patches: {len(train_ds)}  |  Val patches: {len(val_ds)}")

    class_names = get_class_names(cfg)
    num_classes  = get_num_classes(cfg)
    weights = compute_class_weights(train_ds.patches, num_classes).to(device)
    print("Class weights:", {class_names[i]: f"{w:.3f}" for i, w in enumerate(weights)})

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], pin_memory=True)

    model = build_model(cfg).to(device)
    model_desc = (f"smp_unet/{cfg.get('encoder_name','resnet34')}({cfg.get('encoder_weights','imagenet')})"
                  if cfg.get("model") == "smp_unet" else
                  f"custom_unet(base={cfg.get('base_features',64)})")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {model_desc}  |  {n_params:.1f}M params")

    loss_type = cfg.get("loss", "ce")
    if loss_type == "combo":
        criterion = ComboLoss(ce_weight=weights, dice_weight=cfg.get("dice_weight", 0.5))
        print(f"Loss: ComboLoss  (CE weight={1-cfg.get('dice_weight',0.5):.2f}  "
              f"Dice weight={cfg.get('dice_weight',0.5):.2f})")
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)
        print("Loss: CrossEntropyLoss")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "metrics.csv"
    csv_file = open(csv_path, "w", newline="")
    iou_fields = [f"iou_{n}" for n in class_names]
    writer = csv.DictWriter(csv_file,
                            fieldnames=["epoch", "train_loss", "val_loss"] + iou_fields)
    writer.writeheader()

    best_val_loss = float("inf")
    best_fg_iou   = -1.0

    for epoch in range(1, cfg["epochs"] + 1):
        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)

        # ── validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        all_preds, all_tgts = [], []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                val_loss += criterion(logits, masks).item() * imgs.size(0)
                all_preds.append(logits.argmax(1).cpu())
                all_tgts.append(masks.cpu())
        val_loss /= len(val_ds)

        ious    = iou_per_class(torch.cat(all_preds), torch.cat(all_tgts), num_classes)
        iou_str = "  ".join(f"{class_names[c]}={ious[c]:.3f}" for c in range(num_classes))
        fg_iou  = float(np.nanmean(ious[1:]))

        scheduler.step()

        print(f"Epoch {epoch:3d}/{cfg['epochs']} | "
              f"train={train_loss:.4f} | val={val_loss:.4f} | IoU: {iou_str}")

        row = {"epoch": epoch,
               "train_loss": f"{train_loss:.6f}",
               "val_loss":   f"{val_loss:.6f}"}
        row.update({f"iou_{class_names[c]}": f"{ious[c]:.6f}" for c in range(num_classes)})
        writer.writerow(row)
        csv_file.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_loss": val_loss, "fg_iou": fg_iou, "cfg": cfg},
                       out_dir / "best_loss_model.pth")

        if fg_iou > best_fg_iou:
            best_fg_iou = fg_iou
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_loss": val_loss, "fg_iou": fg_iou, "cfg": cfg},
                       out_dir / "best_model.pth")

        if epoch % cfg.get("save_every", 10) == 0:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss, "cfg": cfg},
                       out_dir / f"checkpoint_epoch{epoch:04d}.pth")

    csv_file.close()
    print(f"\nDone.")
    print(f"  Best fg IoU  = {best_fg_iou:.4f}  → {out_dir / 'best_model.pth'}")
    print(f"  Best val loss= {best_val_loss:.4f}  → {out_dir / 'best_loss_model.pth'}")

    print("Plotting training curves...")
    plot_curves(csv_path, out_dir / "training_curves.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/run02_smp_resnet34_3cls_ce_random.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)
