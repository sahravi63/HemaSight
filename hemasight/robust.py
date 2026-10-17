"""Camera-robustness helpers shared by the training scripts.

Different phones render colour slightly differently. A model trained on one
set of cameras can lose a lot of accuracy on another (measured: AUC 0.94 ->
0.79 for a +/-4 % per-channel colour shift). We therefore:
  * train on extra copies of every photo with random per-channel colour gains
    ("colour jitter"), and
  * test every model on simulated new cameras (+/-4 % and +/-8 % shifts).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .features import extract_features


def _one(args):
    path, gains = args
    try:
        return extract_features(path, gains)
    except Exception:
        return None


def jitter_gains(n: int, sd: float, seed: int) -> np.ndarray:
    r = np.random.default_rng(seed)
    return np.clip(r.normal(1.0, sd, size=(n, 3)), 0.7, 1.3)


def features_with_gains(paths, gains, cols, workers=None) -> np.ndarray:
    """Colour features of each photo after applying per-photo colour gains."""
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_one, zip(paths, gains), chunksize=16))
    out = np.full((len(rows), len(cols)), np.nan, dtype=np.float32)
    for i, r in enumerate(rows):
        if r:
            out[i] = [r.get(c, np.nan) for c in cols]
    return out
