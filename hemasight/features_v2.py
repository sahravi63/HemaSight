"""Extra features aimed at generalising across cameras and lighting.

1. Specular-highlight colour constancy: the conjunctiva is moist, so its
   brightest pixels are mirror-like reflections of the light source. Their
   colour estimates the illuminant, which we divide out (von Kries) before
   measuring redness. A grey-edge estimate is added as a second opinion.
2. Spatial layout: colour at the rim of the region vs its interior.
3. Texture: fine red detail (vessels) from a high-pass of the a* channel.
"""
from __future__ import annotations

import numpy as np

from .features import load_rgb, rgb_to_lab, tissue_mask, _stats


def _erode(mask: np.ndarray, it: int) -> np.ndarray:
    m = mask.copy()
    for _ in range(it):
        m = m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
    return m


def _box_blur(x: np.ndarray, r: int = 2) -> np.ndarray:
    k = 2 * r + 1
    p = np.pad(x, r, mode="edge")
    c = p.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def _colour_block(px: np.ndarray, prefix: str) -> dict:
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    s = r + g + b + 1e-6
    lab = rgb_to_lab(np.clip(px, 0, 1))
    out = {}
    out.update(_stats(r / s, f"{prefix}rn"))
    out.update(_stats(g / s, f"{prefix}gn"))
    out.update(_stats(lab[:, 1], f"{prefix}a"))
    out.update(_stats(lab[:, 0], f"{prefix}L"))
    out.update(_stats(np.log10((r + 1e-3) / (g + 1e-3)), f"{prefix}EI"))
    out[f"{prefix}a_over_b"] = float(np.median(lab[:, 1]) / (abs(np.median(lab[:, 2])) + 1.0))
    return out


def extract_features_v2(path: str) -> dict:
    rgba = load_rgb(path)
    mask = tissue_mask(rgba)
    rgb = rgba[..., :3]
    px = rgb[mask].reshape(-1, 3)
    if len(px) < 30:
        raise ValueError("too few tissue pixels")
    f = {}

    # 1a. specular highlights -> illuminant colour
    v = px.max(axis=1)
    hi = px[v >= np.percentile(v, 98)]
    illum = hi.mean(axis=0) / (hi.mean() + 1e-6)
    f.update({"ill_r": float(illum[0]), "ill_g": float(illum[1]), "ill_b": float(illum[2]),
              "ill_rg": float(illum[0] / (illum[1] + 1e-6))})
    px_wb = np.clip(px / (illum + 1e-6) * (px.mean() / (px / (illum + 1e-6)).mean()), 0, 1)
    f.update(_colour_block(px_wb, "wb_"))

    # 1b. grey-edge illuminant (mean gradient colour)
    gx = np.abs(np.diff(rgb, axis=1))[mask[:, 1:] & mask[:, :-1]]
    if len(gx) > 10:
        ge = gx.mean(axis=0) / (gx.mean() + 1e-6)
        f.update({"ge_r": float(ge[0]), "ge_g": float(ge[1]), "ge_b": float(ge[2])})
        px_ge = np.clip(px / (ge + 1e-6) * (px.mean() / (px / (ge + 1e-6)).mean()), 0, 1)
        f.update(_colour_block(px_ge, "ge_"))

    # 2. rim vs interior
    inner = _erode(mask, 3)
    rim = mask & ~inner
    if inner.sum() > 20 and rim.sum() > 20:
        li = rgb_to_lab(rgb[inner]); lr = rgb_to_lab(rgb[rim])
        f["a_inner_minus_rim"] = float(np.median(li[:, 1]) - np.median(lr[:, 1]))
        f["L_inner_minus_rim"] = float(np.median(li[:, 0]) - np.median(lr[:, 0]))

    # 3. texture of redness (vessels)
    lab_img = np.zeros(mask.shape, dtype=np.float32)
    lab_img[mask] = rgb_to_lab(px)[:, 1]
    fill = np.where(mask, lab_img, np.median(lab_img[mask]))
    hp = fill - _box_blur(fill, 2)
    f["a_hp_std"] = float(hp[inner if inner.sum() > 20 else mask].std())
    f["a_hp_p90"] = float(np.percentile(np.abs(hp[mask]), 90))
    lum = rgb.mean(axis=2)
    lfill = np.where(mask, lum, np.median(lum[mask]))
    f["L_hp_std"] = float((lfill - _box_blur(lfill, 2))[mask].std())
    f["shape_fill"] = float(mask.mean())
    return f
