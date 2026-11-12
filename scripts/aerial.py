"""Aerial photo mosaic at GROUND_RES, with row 0 at the north edge."""

import numpy as np
from PIL import Image

from config import BUILD, CHUNK, GROUND_RES, NX, NY, RAW, XMIN, YMAX


def load_aerial():
    """The mosaic as a read-only memory map (uint8, rows north first): a town's photo is over a gigabyte, and most
    scripts only look at parts of it."""
    cache = BUILD / "aerial.npy"
    if not cache.exists():
        px = int(CHUNK / GROUND_RES)
        BUILD.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp.npy")
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(NY * px, NX * px, 3))
        for ix in range(NX):
            for iy in range(NY):
                img = np.array(Image.open(RAW / "aerial" / f"{ix}_{iy}.jpg").convert("RGB"))
                row = (NY - 1 - iy) * px
                out[row : row + px, ix * px : ix * px + px] = img
        out.flush()
        del out
        tmp.rename(cache)
    return np.load(cache, mmap_mode="r")


def world_to_px(xy):
    """RD meters (N,2) to fractional pixel coordinates (col, row) in the mosaic."""
    xy = np.asarray(xy, dtype=np.float64)
    return np.stack([(xy[:, 0] - XMIN) / GROUND_RES, (YMAX - xy[:, 1]) / GROUND_RES], axis=-1)
