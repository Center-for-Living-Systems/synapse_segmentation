import torch.nn as nn

from data.dataset import (CLASS_NAMES, CLASS_NAMES_BINARY,
                          NUM_CLASSES, NUM_CLASSES_BINARY)
from models.unet import UNet


def get_num_classes(cfg: dict) -> int:
    return NUM_CLASSES_BINARY if cfg.get("binary") else NUM_CLASSES


def get_class_names(cfg: dict) -> list[str]:
    return CLASS_NAMES_BINARY if cfg.get("binary") else CLASS_NAMES


def build_model(cfg: dict) -> nn.Module:
    """Instantiate the model described by cfg.

    Supported values of cfg["model"]:
        "smp_unet"     — segmentation-models-pytorch UNet with a pretrained encoder.
                         cfg["encoder_name"] and cfg["encoder_weights"] control the backbone.
        "custom_unet"  — our from-scratch UNet (models/unet.py).
                         cfg["base_features"] controls the channel width.
    """
    num_classes = get_num_classes(cfg)
    if cfg.get("model", "custom_unet") == "smp_unet":
        import segmentation_models_pytorch as smp
        return smp.Unet(
            encoder_name=cfg.get("encoder_name", "resnet34"),
            encoder_weights=cfg.get("encoder_weights", "imagenet"),
            in_channels=1,
            classes=num_classes,
        )
    return UNet(in_channels=1, num_classes=num_classes,
                base_features=cfg.get("base_features", 64))
