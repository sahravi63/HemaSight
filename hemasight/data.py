"""Find images, infer body site and label from folder names, and group
augmented copies of the same photo so splits are leakage-free."""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .features import invariant_signature

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def infer_site(path: str) -> str:
    p = path.lower()
    if "conjunctiv" in p or "eye" in p:
        return "eye"
    if "finger" in p or "nail" in p:
        return "nail"
    if "palm" in p:
        return "palm"
    return "unknown"


_NON = re.compile(r"non[\s_-]*an")           # non-anemic, Non-Anrmic (dataset typo), non anaemic
_ANE = re.compile(r"an(a)?e?r?mi")             # anemic, anaemic, anrmic


def _label_from_text(s: str):
    s = s.lower()
    if _NON.search(s) or "normal" in s or "healthy" in s:
        return 0
    if _ANE.search(s):
        return 1
    return None


def infer_label(path: str) -> int | None:
    """1 = anaemic, 0 = non-anaemic, None = unknown.

    The file name wins (e.g. 'Non-Anrmic-FN-090 (5).png'); otherwise the
    closest folder whose name is a label (not a dataset title)."""
    lab = _label_from_text(Path(path).stem)
    if lab is not None:
        return lab
    for part in reversed(Path(path).parts[:-1]):
        p = part.lower()
        if any(w in p for w in ("dataset", "detect", "application")):
            continue
        lab = _label_from_text(p)
        if lab is not None:
            return lab
    return None


_AUG = re.compile(r"\s*\(\d+\)$")
_NUM = re.compile(r"\d+")


def photo_id(path: str) -> str:
    """Id of the original photo, shared by all its augmented copies.

    'Anemic-FN-108 (7)' -> '108', 'Anemic-001FV (2)' -> '1', 'AnemicP125' -> '125'.
    Grouping is by site + label + number, which is deliberately conservative:
    if two differently named series reuse a number they are kept together,
    which can only make the test folds stricter, never leak."""
    stem = _AUG.sub("", Path(path).stem).strip().lower()
    m = _NUM.search(stem)
    return str(int(m.group())) if m else stem


def discover(root: str) -> pd.DataFrame:
    rows = []
    for dirpath, _, files in os.walk(root, followlinks=True):
        for f in files:
            if Path(f).suffix.lower() in IMG_EXT and not f.startswith("._"):
                full = os.path.join(dirpath, f)
                rel = os.path.relpath(full, root)
                rows.append({"path": full, "rel": rel,
                             "site": infer_site(rel), "label": infer_label(rel)})
    return pd.DataFrame(rows)


class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def assign_groups(df: pd.DataFrame, tol: float = 0.004, use_colour: bool = False) -> pd.Series:
    """Give every image a group id so copies of one photo share it.

    Primary key: the original photo id from the file name ('X (3).png' -> 'X').
    With use_colour=True, images whose flip/rotation-invariant colour
    signatures are nearly identical are also merged (catches copies that
    were renamed)."""
    keys = (df["site"].astype(str) + "|" + df["label"].astype(str) + "|"
            + df["rel"].map(photo_id))
    if not use_colour:
        return pd.Series(keys.values, index=df.index, name="group")

    uf = _UnionFind(len(df))
    first = {}
    for i, k in enumerate(keys):
        if k in first:
            uf.union(first[k], i)
        else:
            first[k] = i
    sigs = np.stack([invariant_signature(p) for p in df["path"]])
    for _, idx in df.groupby(["site", "label"], dropna=False).groups.items():
        pos = df.index.get_indexer(list(idx))
        S = sigs[pos]
        for a in range(len(pos)):
            d = np.abs(S[a + 1:] - S[a]).max(axis=1)
            for b in np.nonzero(d < tol)[0]:
                uf.union(pos[a], pos[a + 1 + b])
    roots = [uf.find(i) for i in range(len(df))]
    return pd.Series([keys.iloc[r] for r in roots], index=df.index, name="group")
