import zipfile
from pathlib import Path

import numpy as np
import roifile
from PIL import Image, ImageDraw


def load_mask(g_zip: str | None, p_zip: str | None, width: int, height: int) -> np.ndarray:
    """Build a 3-class integer mask from ImageJ ROI zip files.

    Returns:
        np.ndarray of shape (H, W) with dtype uint8.
        0 = Background, 1 = G, 2 = P
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    for label, zip_path in [(1, g_zip), (2, p_zip)]:
        if zip_path is None or not Path(zip_path).exists():
            continue
        layer = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(layer)
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                if not name.lower().endswith(".roi"):
                    continue
                roi = roifile.ImagejRoi.frombytes(z.read(name))
                coords = roi.coordinates()
                if coords is not None and len(coords) >= 3:
                    pts = [(float(c[0]), float(c[1])) for c in coords]
                    draw.polygon(pts, fill=1)
        mask[np.array(layer) > 0] = label

    return mask
