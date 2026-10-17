"""Colour feature extraction for anaemia screening.

Only numpy + Pillow are used so this runs anywhere, including IBM LinuxONE
(s390x), where heavy deep-learning wheels may not be available.

The idea: haemoglobin makes tissue red. Pale (anaemic) tissue has a lower
red/green contrast, a lower CIELAB a* ("redness") and a higher L* (lightness).
We describe each image with robust statistics of several colour spaces.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

MAX_SIDE = 256  # downscale for speed; colour statistics barely change


def load_rgb(path: str) -> np.ndarray:
    """Load an image as float32 RGB in [0, 1], EXIF-rotated and downscaled."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (0, 0, 0, 0))
        bg.alpha_composite(img)
        img = bg
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    arr = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
    return arr  # H x W x 4 (alpha kept for masking)


def tissue_mask(rgba: np.ndarray) -> np.ndarray:
    """Pixels that belong to tissue.

    The Ghana datasets are already cropped to the region of interest with a
    black (or transparent) background, so we drop transparent, near-black and
    blown-out pixels.
    """
    rgb, alpha = rgba[..., :3], rgba[..., 3]
    v = rgb.max(axis=-1)
    mask = (alpha > 0.5) & (v > 0.06) & (v < 0.995)
    if mask.sum() < 50:  # fall back to everything that is not transparent
        mask = alpha > 0.5
    return mask


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (N x 3, 0..1) -> CIELAB (D65)."""
    lin = srgb_to_linear(rgb)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    xyz = lin @ m.T
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    eps, k = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (k * xyz + 16) / 116)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return np.stack([L, a, b], axis=1)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx, mn = rgb.max(axis=1), rgb.min(axis=1)
    d = mx - mn + 1e-8
    h = np.where(mx == r, ((g - b) / d) % 6,
        np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) / 6.0
    s = np.where(mx > 0, (mx - mn) / (mx + 1e-8), 0)
    # hue is circular and tissue sits around red (0/1); shift so red is continuous
    h = (h + 0.5) % 1.0
    return np.stack([h, s, mx], axis=1)


def _stats(x: np.ndarray, prefix: str) -> dict:
    q = np.percentile(x, [10, 25, 50, 75, 90])
    return {
        f"{prefix}_mean": float(x.mean()), f"{prefix}_std": float(x.std()),
        f"{prefix}_p10": float(q[0]), f"{prefix}_p25": float(q[1]),
        f"{prefix}_p50": float(q[2]), f"{prefix}_p75": float(q[3]),
        f"{prefix}_p90": float(q[4]),
    }


def extract_features(path, gains=None) -> dict:
    """Return a flat dict of colour features for one image.

    `path` may also be an RGBA float array. `gains` (3 per-channel multipliers)
    simulates a different camera's colour rendering, for robustness training."""
    rgba = load_rgb(path) if isinstance(path, str) else path
    if gains is not None:
        rgba = rgba.copy()
        rgba[..., :3] = np.clip(rgba[..., :3] * np.asarray(gains, dtype=np.float32), 0, 1)
    mask = tissue_mask(rgba)
    px = rgba[..., :3][mask].reshape(-1, 3)
    if len(px) == 0:
        raise ValueError(f"no tissue pixels found in {path}")

    feats: dict = {"n_pixels": float(len(px))}
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    s = r + g + b + 1e-6
    lab = rgb_to_lab(px)
    hsv = rgb_to_hsv(px)

    channels = {
        "R": r, "G": g, "B": b,
        "rn": r / s, "gn": g / s, "bn": b / s,          # chromaticity (light-invariant-ish)
        "L": lab[:, 0], "a": lab[:, 1], "bb": lab[:, 2],
        "H": hsv[:, 0], "S": hsv[:, 1], "V": hsv[:, 2],
        # erythema index: log ratio of red to green reflectance, a classic
        # proxy for haemoglobin under the skin / mucosa
        "EI": np.log10((r + 1e-3) / (g + 1e-3)),
        "RG": r - g,
    }
    for name, ch in channels.items():
        feats.update(_stats(ch, name))

    # 8-bin histogram of a* (redness) and of chromatic red
    for name, ch, lo, hi in (("a", lab[:, 1], -10, 60), ("rn", r / s, 0.2, 0.7)):
        hist, _ = np.histogram(np.clip(ch, lo, hi), bins=8, range=(lo, hi))
        hist = hist / max(hist.sum(), 1)
        for i, v in enumerate(hist):
            feats[f"hist_{name}_{i}"] = float(v)

    # grey-world normalised redness: divides out a global colour cast
    gw = px / (px.mean(axis=0, keepdims=True) + 1e-6)
    feats["gw_rg_mean"] = float((gw[:, 0] - gw[:, 1]).mean())
    feats["a_over_L"] = float(lab[:, 1].mean() / (lab[:, 0].mean() + 1e-6))
    return feats


def invariant_signature(path: str) -> np.ndarray:
    """Colour signature that does not change under flips/rotations.

    Used to find augmented copies of the same original photo so they never
    end up on both sides of a train/test split.
    """
    rgba = load_rgb(path)
    px = rgba[..., :3][tissue_mask(rgba)].reshape(-1, 3)
    q = np.percentile(px, [10, 25, 50, 75, 90], axis=0).ravel()
    return np.concatenate([px.mean(0), px.std(0), q]).astype(np.float32)
