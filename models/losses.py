import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Soft Dice loss averaged over foreground classes only (skips background = class 0).

    Why foreground only: background dominates pixel count and would swamp the Dice
    signal for the rare G / P / Cell classes we actually care about.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W)   targets: (B, H, W) integer class labels
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        targets_oh = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1)
        dice_scores = []
        for c in range(1, num_classes):
            p = probs[:, c].reshape(-1)
            t = targets_oh[:, c].reshape(-1)
            inter = (p * t).sum()
            dice_scores.append((2 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth))
        return 1.0 - torch.stack(dice_scores).mean()


class ComboLoss(nn.Module):
    """Weighted sum of class-weighted CrossEntropyLoss and DiceLoss.

    L = (1 - dice_weight) * CE + dice_weight * (1 - Dice_fg)

    Why combo: CE alone picks the easiest minimum (predict all background), so its
    minimum epoch is a poor checkpoint criterion. Adding Dice directly optimises
    overlap on the rare foreground classes and stabilises training.
    """

    def __init__(self, ce_weight: torch.Tensor = None, dice_weight: float = 0.5):
        super().__init__()
        self.ce          = nn.CrossEntropyLoss(weight=ce_weight)
        self.dice        = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (1 - self.dice_weight) * self.ce(logits, targets) + \
                    self.dice_weight  * self.dice(logits, targets)
