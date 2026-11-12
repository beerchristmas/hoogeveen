# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "scikit-image", "tifffile", "imagecodecs", "pillow"]
# ///
"""Find individual trees in the AHN canopy height model.

Output (data/build/):
  trees.bin   float32 per tree: x, z, height, crown radius, crown base, kind, r, g, b, seed
  chm.npy     canopy height model at 0.5 m (row 0 north), used for hedges and shadows
"""

import json

import numpy as np
import tifffile
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from aerial import load_aerial
from config import BUILD, CHUNK, CX, CY, GROUND_RES, NX, NY, RAW, XMIN, YMAX

RES = 0.5


def mosaic(cov, ix0, ix1, iy0, iy1):
    """The chunks ix0..ix1 by iy0..iy1 (ends excluded, iy counting from the south) as one raster, row 0 north."""
    px = int(CHUNK / RES)
    out = np.full(((iy1 - iy0) * px, (ix1 - ix0) * px), np.nan, dtype=np.float32)
    for ix in range(ix0, ix1):
        for iy in range(iy0, iy1):
            a = tifffile.imread(RAW / "ahn" / f"{cov}_{ix}_{iy}.tif").astype(np.float32)
            a[(a > 1000) | (a < -100)] = np.nan
            row = (iy1 - 1 - iy) * px
            out[row : row + px, (ix - ix0) * px : (ix - ix0 + 1) * px] = a[:px, :px]
    return out


def fill_nearest(a):
    bad = np.isnan(a)
    if bad.all():
        return np.zeros_like(a)
    idx = ndi.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def footprint_mask(rings, shape, col0, row0, grow_m):
    """Building footprints in the window that starts at column col0, row row0 of the whole raster."""
    img = Image.new("L", (shape[1], shape[0]), 0)
    d = ImageDraw.Draw(img)
    for ring in rings:
        d.polygon([((x + CX - XMIN) / RES - col0, (YMAX - (CY - z)) / RES - row0) for x, z in ring], fill=255)
    m = np.array(img) > 0
    return ndi.binary_dilation(m, iterations=max(1, int(grow_m / RES)))


# Trees are found in blocks of this many chunks a side, each with the ring of chunks around it as a rim: a tree only
# depends on the canopy close by, and a town's canopy model with everything derived from it does not fit in memory.
BLOCK = 3


def find_trees(ix0, ix1, iy0, iy1, rings, aerial, chm_out, rng):
    """The trees whose tops stand in the chunks ix0..ix1 by iy0..iy1, as rows for trees.bin, with their column and
    row in the whole canopy raster. Writes the block's canopy heights into chm_out."""
    px = int(CHUNK / RES)
    wx0, wx1, wy0, wy1 = max(ix0 - 1, 0), min(ix1 + 1, NX), max(iy0 - 1, 0), min(iy1 + 1, NY)
    col0, row0 = wx0 * px, (NY - wy1) * px  # the window's corner in the whole raster
    dsm = mosaic("dsm_05m", wx0, wx1, wy0, wy1)
    dtm = fill_nearest(mosaic("dtm_05m", wx0, wx1, wy0, wy1))
    chm = np.nan_to_num(dsm - dtm, nan=0.0)
    del dsm, dtm
    chm[chm < 0] = 0
    chm[footprint_mask(rings, chm.shape, col0, row0, 1.0)] = 0
    # The block itself, inside the window.
    c0, c1, r0, r1 = ix0 * px - col0, ix1 * px - col0, (NY - iy1) * px - row0, (NY - iy0) * px - row0
    chm_out[row0 + r0 : row0 + r1, col0 + c0 : col0 + c1] = chm[r0:r1, c0:c1].astype(np.float16)

    smooth = ndi.gaussian_filter(chm, 1.2)
    canopy = smooth > 2.5
    peaks = (smooth == ndi.maximum_filter(smooth, size=5)) & (smooth > 3.5)
    ys, xs = np.nonzero(peaks)
    hs = smooth[ys, xs]

    # Greedy suppression: taller trees claim a radius that grows with their height.
    order = np.argsort(-hs)
    taken = np.zeros_like(peaks)
    keep = []
    for i in order:
        y, x, h = ys[i], xs[i], hs[i]
        if taken[y, x]:
            continue
        keep.append(i)
        r = int(np.clip(0.12 * h + 1.0, 1.5, 4.5) / RES)
        taken[max(0, y - r) : y + r + 1, max(0, x - r) : x + r + 1] = True
    ys, xs, hs = ys[keep], xs[keep], hs[keep]
    del peaks, taken
    if not len(ys):
        return np.zeros((0, 10), np.float32), ys, xs

    index = np.arange(1, len(ys) + 1)
    markers = np.zeros(chm.shape, dtype=np.int32)
    markers[ys, xs] = index
    labels = watershed(-smooth, markers, mask=canopy)
    del markers, smooth, canopy
    areas = np.bincount(labels.ravel(), minlength=len(ys) + 1)[1:] * RES * RES
    radius = np.clip(np.sqrt(areas / np.pi), 1.0, 7.5)

    # Crown color from the aerial photo: evergreens stay green in the (spring) photo. One colour channel at a time,
    # averaged down to the canopy grid.
    scale = int(RES / GROUND_RES)
    cols = []
    for c in range(3):
        a = np.asarray(aerial[row0 * scale : (row0 + chm.shape[0]) * scale, col0 * scale : (col0 + chm.shape[1]) * scale, c]).astype(np.float32)
        small = a.reshape(a.shape[0] // scale, scale, a.shape[1] // scale, scale).mean(axis=(1, 3))
        cols.append(ndi.median(small, labels, index=index))
        del a, small
    cols = np.nan_to_num(np.stack(cols, -1), nan=90.0)
    r, g, b = cols[:, 0], cols[:, 1], cols[:, 2]
    evergreen = (g > r + 4) & (g > b + 4) & (g < 110)

    wx = XMIN + (col0 + xs + 0.5) * RES
    wy = YMAX - (row0 + ys + 0.5) * RES
    base = np.clip(hs * rng.uniform(0.25, 0.4, len(hs)), 1.2, 6)
    trees = np.stack(
        [wx - CX, CY - wy, hs, radius, base, evergreen.astype(np.float32), r / 255, g / 255, b / 255, rng.random(len(hs))], -1
    ).astype(np.float32)
    inside = (xs >= c0) & (xs < c1) & (ys >= r0) & (ys < r1)
    return trees[inside], row0 + ys[inside], col0 + xs[inside]


def main():
    px = int(CHUNK / RES)
    shape = (NY * px, NX * px)
    BUILD.mkdir(parents=True, exist_ok=True)
    chm = np.lib.format.open_memmap(BUILD / "chm.npy", mode="w+", dtype=np.float16, shape=shape)
    rings = json.load(open(BUILD / "footprints.json"))
    aerial = load_aerial()
    parts, ys, xs = [], [], []
    for n, (ix0, iy0) in enumerate((i, j) for i in range(0, NX, BLOCK) for j in range(0, NY, BLOCK)):
        t, y, x = find_trees(ix0, min(ix0 + BLOCK, NX), iy0, min(iy0 + BLOCK, NY), rings, aerial, chm, np.random.default_rng(7 + n))
        parts.append(t)
        ys.append(y)
        xs.append(x)
    trees, ys, xs = np.concatenate(parts), np.concatenate(ys), np.concatenate(xs)
    kind, radius = trees[:, 5], trees[:, 3]
    rng = np.random.default_rng(7)

    # Municipal trees from the BGT that the canopy model missed (young street trees).
    bgt = json.load(open(RAW / "bgt_vegetatieobject_punt.geojson"))["features"]
    pts = np.array([f["geometry"]["coordinates"][:2] for f in bgt]).reshape(-1, 2)
    have = np.zeros(shape, bool)
    have[ys, xs] = True
    have = ndi.binary_dilation(have, iterations=int(3 / RES))
    extra = []
    for x, y in pts:
        col, row = int((x - XMIN) / RES), int((YMAX - y) / RES)
        if not (0 <= col < shape[1] and 0 <= row < shape[0]) or have[row, col]:
            continue
        h = max(float(chm[row, col]), 4.0 + rng.random() * 3)
        extra.append([x - CX, CY - y, h, 1.2 + h * 0.15, 1.5, 0, 0.35, 0.42, 0.25, rng.random()])
    if extra:
        trees = np.concatenate([trees, np.array(extra, dtype=np.float32)])
    # Masts show up in the canopy model as very tall, thin "trees".
    lm = BUILD / "landmarks.json"
    for m in json.loads(lm.read_text()).get("masts", []) if lm.exists() else []:
        trees = trees[np.hypot(trees[:, 0] - (m["x"] - CX), trees[:, 1] - (CY - m["y"])) > 6]
    trees.tofile(BUILD / "trees.bin")
    print(f"{len(trees)} trees ({len(extra)} from BGT), {int(kind.sum())} evergreen, mean radius {radius.mean():.1f} m")


if __name__ == "__main__":
    main()
