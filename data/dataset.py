from pathlib import Path

import numpy as np
import tifffile
import torch
from scipy.ndimage import gaussian_filter, map_coordinates
from torch.utils.data import Dataset

from utils.roi_utils import load_mask

CLASS_NAMES        = ["Background", "G", "P"]
CLASS_NAMES_BINARY = ["Background", "Cell"]
NUM_CLASSES        = 3
NUM_CLASSES_BINARY = 2


# ── numpy-level augmentation ──────────────────────────────────────────────────

def _elastic_deform(
    img: np.ndarray,
    mask: np.ndarray,
    alpha: float = 80.0,
    sigma: float = 10.0,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Random elastic deformation applied identically to image and mask."""
    if rng is None:
        rng = np.random.default_rng()
    H, W = img.shape
    dx = gaussian_filter(rng.standard_normal((H, W)), sigma) * alpha
    dy = gaussian_filter(rng.standard_normal((H, W)), sigma) * alpha
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    coords = [np.clip(y + dy, 0, H - 1), np.clip(x + dx, 0, W - 1)]
    img_d  = map_coordinates(img,  coords, order=1, mode="reflect").astype(np.float32)
    mask_d = map_coordinates(mask.astype(np.float32), coords, order=0, mode="reflect").astype(mask.dtype)
    return img_d, mask_d


def _intensity_jitter(
    img: np.ndarray,
    brightness: float = 0.25,
    contrast: float = 0.25,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Random brightness and contrast shift on a float32 image in [0, 1]."""
    if rng is None:
        rng = np.random.default_rng()
    b = 1.0 + rng.uniform(-brightness, brightness)
    c = 1.0 + rng.uniform(-contrast, contrast)
    mean = img.mean()
    img = img * b
    img = (img - mean) * c + mean
    return np.clip(img, 0.0, 1.0).astype(np.float32)


# ── patch sampling ────────────────────────────────────────────────────────────

def _sample_patches(
    img: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
    n_patches: int,
    rng: np.random.Generator,
    annotated_fraction: float = 0.0,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[int, int]]]:
    """Sample patches from (img, mask).

    annotated_fraction: fraction of patches guaranteed to contain at least one
    annotated pixel (mask > 0).  0 = pure random (original behaviour).
    """
    H, W = img.shape
    if H < patch_size or W < patch_size:
        img  = np.pad(img,  ((0, max(0, patch_size - H)), (0, max(0, patch_size - W))))
        mask = np.pad(mask, ((0, max(0, patch_size - H)), (0, max(0, patch_size - W))))
        H, W = img.shape

    max_y, max_x = H - patch_size, W - patch_size

    if annotated_fraction > 0 and mask.max() > 0:
        # Integral image to find which positions overlap any annotated pixel
        ann  = (mask > 0).astype(np.int64)
        pad  = np.zeros((H + 1, W + 1), dtype=np.int64)
        pad[1:, 1:] = ann
        cs   = pad.cumsum(axis=0).cumsum(axis=1)
        YY, XX = np.mgrid[0 : max_y + 1, 0 : max_x + 1]
        sums = (cs[YY + patch_size, XX + patch_size]
                - cs[YY,            XX + patch_size]
                - cs[YY + patch_size, XX           ]
                + cs[YY,            XX             ])
        ann_pos = np.argwhere(sums > 0)   # (N_ann, 2)
        bg_pos  = np.argwhere(sums == 0)  # (N_bg, 2)

        n_ann = min(int(np.ceil(n_patches * annotated_fraction)), len(ann_pos))
        n_bg  = n_patches - n_ann

        sel = ann_pos[rng.choice(len(ann_pos), n_ann, replace=True)]
        ys, xs = list(sel[:, 0]), list(sel[:, 1])

        if n_bg > 0:
            pool = bg_pos if len(bg_pos) > 0 else ann_pos
            sel2 = pool[rng.choice(len(pool), n_bg, replace=True)]
            ys += list(sel2[:, 0])
            xs += list(sel2[:, 1])
    else:
        ys = rng.integers(0, max_y + 1, n_patches).tolist()
        xs = rng.integers(0, max_x + 1, n_patches).tolist()

    patches   = [(img [y : y + patch_size, x : x + patch_size].copy(),
                  mask[y : y + patch_size, x : x + patch_size].copy())
                 for y, x in zip(ys, xs)]
    positions = [(int(y), int(x)) for y, x in zip(ys, xs)]
    return patches, positions


# ── image loading ─────────────────────────────────────────────────────────────

def load_all_images(
    data_dir: str,
    patch_size: int = 256,
    patches_per_image: int = 100,
    seed: int = 42,
    annotated_fraction: float = 0.0,
) -> list[dict]:
    """Load every TIF+ROI pair in data_dir.

    Returns a list of dicts (one per file):
        name, image, mask, patches, positions, image_shape
    """
    rng = np.random.default_rng(seed)
    records = []

    for tif_path in sorted(Path(data_dir).glob("*F.tif")):
        stem  = tif_path.stem[:-1]
        g_zip = Path(data_dir) / f"{stem}Gs.zip"
        p_zip = Path(data_dir) / f"{stem}Ps.zip"

        raw = tifffile.imread(str(tif_path)).astype(np.float32)
        img = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)

        H, W = img.shape
        mask = load_mask(
            str(g_zip) if g_zip.exists() else None,
            str(p_zip) if p_zip.exists() else None,
            W, H,
        )

        patches, positions = _sample_patches(
            img, mask, patch_size, patches_per_image, rng, annotated_fraction)

        print(f"  {tif_path.name}: {H}×{W}  G={g_zip.exists()}  P={p_zip.exists()}  "
              f"patches={len(patches)}")

        records.append({
            "name": tif_path.stem, "image": img, "mask": mask,
            "patches": patches, "positions": positions,
            "image_shape": (H, W),
        })

    return records


# ── dataset ───────────────────────────────────────────────────────────────────

class AugTransform:
    """Tensor-level spatial augmentation: random flip + 90° rotation."""
    def __call__(self, img: torch.Tensor, mask: torch.Tensor):
        if torch.rand(1) > 0.5:
            img,  mask = torch.flip(img,  [-1]), torch.flip(mask, [-1])
        if torch.rand(1) > 0.5:
            img,  mask = torch.flip(img,  [-2]), torch.flip(mask, [-2])
        k = torch.randint(4, (1,)).item()
        if k:
            img  = torch.rot90(img,  k, [-2, -1])
            mask = torch.rot90(mask, k, [-2, -1])
        return img, mask


class PatchDataset(Dataset):
    def __init__(
        self,
        patches: list[tuple[np.ndarray, np.ndarray]],
        augment: bool = False,
        binary: bool = False,
        aug_elastic: bool = True,
        aug_intensity: bool = True,
    ):
        self.patches      = patches
        self.augment      = augment
        self.binary       = binary
        self.aug_elastic  = aug_elastic
        self.aug_intensity = aug_intensity
        self.transform    = AugTransform() if augment else None

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, mask = self.patches[idx]

        # numpy-level augmentation (operates on float32 arrays)
        if self.augment:
            rng = np.random.default_rng()
            if self.aug_elastic and rng.random() > 0.5:
                img, mask = _elastic_deform(img, mask, rng=rng)
            if self.aug_intensity:
                img = _intensity_jitter(img, rng=rng)

        img_t  = torch.from_numpy(img.copy()).unsqueeze(0)
        mask_t = torch.from_numpy(mask.astype(np.int64))
        if self.binary:
            mask_t = (mask_t > 0).long()

        # tensor-level augmentation (flip + rotation)
        if self.transform is not None:
            img_t, mask_t = self.transform(img_t, mask_t)

        return img_t, mask_t


def make_train_val_datasets(
    data_dir: str,
    val_image_idx: int = 3,
    patch_size: int = 256,
    patches_per_image: int = 100,
    seed: int = 42,
    binary: bool = False,
    annotated_fraction: float = 0.0,
    aug_elastic: bool = True,
    aug_intensity: bool = True,
) -> tuple["PatchDataset", "PatchDataset"]:
    """Image-level split: one full image held out for validation."""
    print("Loading images and extracting patches...")
    images = load_all_images(
        data_dir, patch_size, patches_per_image, seed, annotated_fraction)

    train_patches = [p for i, d in enumerate(images) for p in d["patches"] if i != val_image_idx]
    val_patches   = [p for i, d in enumerate(images) for p in d["patches"] if i == val_image_idx]

    val_name = images[val_image_idx]["name"] if val_image_idx < len(images) else "?"
    print(f"  Train patches: {len(train_patches)} | Val patches: {len(val_patches)} (from {val_name})")

    return (
        PatchDataset(train_patches, augment=True,  binary=binary,
                     aug_elastic=aug_elastic, aug_intensity=aug_intensity),
        PatchDataset(val_patches,   augment=False, binary=binary),
    )
