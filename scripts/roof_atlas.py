"""Real roof detail from the aerial photo: solar panels, skylights, chimneys, moss and patched tiles.

For every building the photo is first lined up with the footprint (the photo leans and is never exactly on top of
the BAG outline), then each roof face is brought to the same brightness so the sun and shade sides of the photo
don't double up with the renderer's own lighting. The crops are packed into one texture atlas.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from aerial import world_to_px
from config import BUILD, GROUND_RES

SEARCH_M = 2.0  # how far the photo may be off, in meters

# Same palette as ROOFS in render/scene.js: the main color of every roof snaps to one of these.
PALETTE = np.array([
    (176, 84, 58), (140, 64, 50), (108, 56, 46), (66, 68, 74), (88, 90, 96), (120, 122, 124),
    (150, 150, 146), (184, 182, 176), (120, 92, 70), (98, 114, 98), (214, 214, 206), (44, 48, 56),
], np.float32)


def quantize(img, covered, fixed=None, k=4):
    """Reduce a roof to a few flat colors, like a pixel artist would: the main color from the palette (or a fixed
    color for monuments), the rest (solar panels, skylights, chimneys) kept as their own, slightly punchier colors."""
    px = img[covered]
    k = min(k, max(1, len(px) // 40))
    lum = px.mean(1)
    order = np.argsort(lum)
    centers = np.stack([px[order[int((i + 0.5) / k * (len(px) - 1))]] for i in range(k)]).astype(np.float32)
    for _ in range(8):
        lab = ((px[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if (lab == j).any():
                centers[j] = px[lab == j].mean(0)
    labels = np.full(covered.shape, -1, np.int32)
    labels[covered] = lab
    # Clean up single-pixel speckle.
    labels = np.where(covered, ndi.median_filter(np.where(covered, labels, 0), size=3), -1)
    counts = np.bincount(labels[covered], minlength=k)
    main = int(counts.argmax())
    out = np.empty_like(centers)
    if fixed is not None:
        target = np.array(fixed, np.float32)
    else:
        sat = centers[main].mean() + (centers[main] - centers[main].mean()) * 1.5
        target = PALETTE[((PALETTE - sat) ** 2 * (0.9, 1.2, 1.0)).sum(1).argmin()]
    shift = target - centers[main]
    for j in range(k):
        c = centers[j] + shift  # keep each feature's difference from the main roof color
        m = c.mean()
        out[j] = np.clip(m + (c - m) * 1.3, 0, 255)
    out[main] = target
    res = np.empty_like(img)
    res[:] = target
    res[covered] = out[labels[covered]]
    return res
PAGE = 8192  # side of an atlas page in pixels: the largest texture the smaller GPUs and software renderers take
PAD = 2  # atlas padding in pixels, so neighbouring crops don't bleed into each other


def gradient(aerial, rows=2000):
    """Edge strength of the photo, worked out in strips of rows so the whole mosaic never sits in memory as floats."""
    h, w = aerial.shape[:2]
    out = np.empty((h, w), np.float16)
    for r0 in range(0, h, rows):
        a, b = max(r0 - 2, 0), min(r0 + rows + 2, h)
        lum = np.asarray(aerial[a:b]).astype(np.float32).mean(-1)
        g = np.hypot(ndi.sobel(lum, 0), ndi.sobel(lum, 1))
        out[r0 : min(r0 + rows, h)] = g[r0 - a : r0 - a + min(rows, h - r0)]
    return out


def outline_points(ring, step=GROUND_RES):
    pts = []
    for a, b in zip(ring, np.roll(ring, -1, axis=0)):
        n = max(1, int(np.hypot(*(b - a)) / step))
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        pts.append(a + (b - a) * t)
    return np.concatenate(pts)


def best_shift(grad, ring):
    """Shift in meters (RD x, y) that puts the footprint outline on the strongest edges in the photo."""
    px = world_to_px(outline_points(np.asarray(ring, dtype=np.float64)[:, :2]))
    h, w = grad.shape
    steps = np.arange(-SEARCH_M, SEARCH_M + 1e-6, GROUND_RES)
    best, best_score, zero_score = (0.0, 0.0), -1.0, None
    for sx in steps:
        for sy in steps:
            c = np.clip(np.round(px[:, 0] + sx / GROUND_RES).astype(int), 0, w - 1)
            r = np.clip(np.round(px[:, 1] - sy / GROUND_RES).astype(int), 0, h - 1)
            score = float(grad[r, c].mean()) - 0.6 * np.hypot(sx, sy)  # prefer small shifts
            if sx == 0 and sy == 0:
                zero_score = score
            if score > best_score:
                best, best_score = (sx, sy), score
    if zero_score is not None and best_score < zero_score * 1.08:
        return 0.0, 0.0
    return best


def build(aerial, roofs, fixed=None):
    """roofs: list of (key, footprint ring, [roof triangles (M,3,3) in RD per face]); fixed: {key: main roof color}.

    Returns {key: (shift, (x, y) on its atlas page, crop origin in mosaic px, page)}, the page size and the number of
    pages, and writes roof_atlas.png (and roof_atlas_1.png, ... for further pages)."""
    grad = gradient(aerial)
    H, W = aerial.shape[:2]
    crops = []
    for key, ring, faces in roofs:
        if not faces:
            continue
        sx, sy = best_shift(grad, ring)
        tris = np.concatenate(faces)
        pts = world_to_px(tris[..., :2].reshape(-1, 2) + (sx, sy))
        c0, r0 = np.floor(pts.min(0)).astype(int) - 1
        c1, r1 = np.ceil(pts.max(0)).astype(int) + 2
        c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, W), min(r1, H)
        if c1 - c0 < 2 or r1 - r0 < 2:
            continue
        crop = aerial[r0:r1, c0:c1].astype(np.float32)
        lum = crop.mean(-1)
        masks = []
        for f in faces:
            m = Image.new("L", (c1 - c0, r1 - r0), 0)
            d = ImageDraw.Draw(m)
            for t in world_to_px(f[..., :2].reshape(-1, 2) + (sx, sy)).reshape(-1, 3, 2):
                d.polygon([(float(x - c0), float(y - r0)) for x, y in t], fill=1)
            masks.append(np.array(m) > 0)
        covered = np.any(masks, axis=0) if masks else np.zeros(lum.shape, bool)
        if covered.sum() < 4:
            continue
        target = float(np.median(lum[covered]))
        out = crop.copy()
        for m in masks:
            if m.sum() < 4:
                continue
            face_lum = float(np.median(lum[m]))
            out[m] = crop[m] * (target / max(face_lum, 1.0))
        out = quantize(out, covered, (fixed or {}).get(key))
        crops.append((key, (sx, sy), (c0, r0), out.astype(np.uint8)))

    # Shelf packing, tallest first, onto pages of at most PAGE x PAGE: a town's roofs do not fit in one texture that
    # every GPU takes. A single page is cut down to the power of two that holds it.
    crops.sort(key=lambda c: -c[3].shape[0])
    x = y = shelf = page = 0
    placed, heights = [], []
    for key, shift, origin, img in crops:
        h, w = img.shape[:2]
        if x + w + PAD > PAGE:
            x, y, shelf = 0, y + shelf + PAD, 0
        if y + h + PAD > PAGE:
            heights.append(y)
            x = y = shelf = 0
            page += 1
        placed.append((key, shift, origin, img, x, y, page))
        x += w + PAD
        shelf = max(shelf, h)
    heights.append(y + shelf + PAD)
    size = (PAGE, PAGE if page else int(2 ** np.ceil(np.log2(heights[0]))))
    pages = [np.zeros((size[1], size[0], 3), np.uint8) for _ in heights]
    rects = {}
    for key, shift, origin, img, x, y, k in placed:
        h, w = img.shape[:2]
        atlas = pages[k]
        atlas[y : y + h, x : x + w] = img
        # Repeat the edge pixels into the padding.
        atlas[y : y + h, x + w : x + w + PAD] = img[:, -1:]
        atlas[y + h : y + h + PAD, x : x + w] = img[-1:, :]
        rects[key] = (shift, (x, y), origin, k)
    for f in BUILD.glob("roof_atlas*.png"):
        f.unlink()
    for k, atlas in enumerate(pages):
        Image.fromarray(atlas).save(BUILD / ("roof_atlas.png" if k == 0 else f"roof_atlas_{k}.png"))
    return rects, size, len(pages)


def atlas_uv(rd_xy, rect, size):
    """Atlas texture coordinates (u right, v up as in WebGL with flipY) for RD points on a roof. The page number is
    added to u: render/scene.js takes it off again and gives every page its own mesh."""
    (sx, sy), (ax, ay), (c0, r0), page = rect
    px = world_to_px(np.asarray(rd_xy) + (sx, sy))
    u = page + np.clip((ax + px[:, 0] - c0) / size[0], 0.0, 0.9999)
    v = 1.0 - (ay + px[:, 1] - r0) / size[1]
    return u, v
