"""Train HemaSight per-site anaemia models.

Usage:
    python3 train.py --data ~/data/IBM_Zthon --out models

Steps
  1. find images, infer body site (eye / nail / palm) and label from folders
  2. extract colour features (cached to features.csv)
  3. group augmented copies of the same photo (leakage-free splits)
  4. 5-fold grouped, stratified cross-validation of several models per site
  5. train with colour-jittered copies of every photo (camera robustness)
     and test each model on simulated new cameras (+/-4 %, +/-8 % colour)
  6. pick the model with the best average AUC across normal and new-camera
     conditions, choose a threshold that keeps sensitivity >= 90 %,
     refit on all data and save it
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import joblib
import numpy as np
import pandas as pd

from hemasight.data import assign_groups, discover
from hemasight.features import extract_features
from hemasight.models import (binary_metrics, candidate_models,
                              stratified_group_folds, threshold_for_sensitivity)
from hemasight.robust import features_with_gains, jitter_gains


def _feat(path):
    try:
        return extract_features(path)
    except Exception as e:  # unreadable image: skip, but report
        return {"_error": str(e)}


META = ["path", "rel", "site", "label", "group"]


def build_table(root: str, cache: str, workers: int) -> pd.DataFrame:
    """Image list + colour features. Features are cached by relative path;
    site, label and group are always recomputed so labelling fixes apply."""
    df = discover(root)
    if df.empty:
        raise SystemExit(f"no images found under {root}")
    print(f"found {len(df)} images")
    print(df.groupby(["site", "label"], dropna=False).size().rename("n").to_string(), "\n")

    feats = None
    if os.path.exists(cache):
        cached = pd.read_csv(cache)
        if set(df["rel"]) <= set(cached["rel"]):
            print(f"using cached features from {cache}")
            feats = cached.drop(columns=[c for c in META if c != "rel"], errors="ignore")
    if feats is None:
        t = time.time()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(_feat, df["path"], chunksize=16))
        F = pd.DataFrame(rows)
        F.insert(0, "rel", df["rel"].values)
        if "_error" in F:
            bad = F["_error"].notna()
            if bad.any():
                print(f"skipped {int(bad.sum())} unreadable images")
            F = F[~bad].drop(columns=["_error"])
        print(f"features: {F.shape[1] - 1} per image, {time.time() - t:.1f}s")
        F.to_csv(cache, index=False)
        feats = F

    out = df.merge(feats, on="rel", how="inner").reset_index(drop=True)
    out["group"] = assign_groups(out).values
    return out


def train_site(site: str, d: pd.DataFrame, feat_cols: list, out_dir: str, k: int,
               jitter_sd: float, jitter_copies: int, workers: int) -> dict:
    d = d[d["label"].notna()].reset_index(drop=True)
    y = d["label"].astype(int).values
    X = d[feat_cols].values.astype(np.float32)
    groups = d["group"].values
    n, n_groups = len(d), len(np.unique(groups))
    print(f"\n=== {site}: {n} images, {n_groups} unique photos after grouping "
          f"({int(y.sum())} anaemic / {int((1 - y).sum())} non-anaemic images)")

    # colour-jittered training copies and two simulated new cameras (see hemasight/robust.py)
    paths = d["path"].tolist()
    Xjit = [features_with_gains(paths, jitter_gains(n, jitter_sd, 200 + j), feat_cols, workers)
            for j in range(jitter_copies)]
    Xcam = {"camera_4pct": features_with_gains(paths, jitter_gains(n, 0.04, 997), feat_cols, workers),
            "camera_8pct": features_with_gains(paths, jitter_gains(n, 0.08, 996), feat_cols, workers)}
    for Xa in Xjit + list(Xcam.values()):
        bad = np.isnan(Xa).any(axis=1)
        Xa[bad] = X[bad]

    def rows(idx):
        return np.vstack([X[idx]] + [Xj[idx] for Xj in Xjit]), np.tile(y[idx], 1 + len(Xjit))

    results, oof_all = {}, {}
    for name, model in candidate_models().items():
        if name == "logreg":
            continue
        P = {"normal": np.zeros(n), **{c: np.zeros(n) for c in Xcam}}
        try:
            for tr, te in stratified_group_folds(y, groups, k=k):
                m = _fresh(model)
                m.fit(*rows(tr))
                P["normal"][te] = m.predict_proba(X[te])[:, 1]
                for c, Xc in Xcam.items():
                    P[c][te] = m.predict_proba(Xc[te])[:, 1]
        except Exception as e:
            print(f"  {name:14s} failed: {e}")
            continue
        met = binary_metrics(y, P["normal"], 0.5)
        met["auc_camera_4pct"] = binary_metrics(y, P["camera_4pct"], 0.5)["auc"]
        met["auc_camera_8pct"] = binary_metrics(y, P["camera_8pct"], 0.5)["auc"]
        met["robust_auc"] = round(float(np.mean([met["auc"], met["auc_camera_4pct"], met["auc_camera_8pct"]])), 4)
        results[name], oof_all[name] = met, P["normal"]
        print(f"  {name:14s} AUC {met['auc']:.3f} | new camera ±4%: {met['auc_camera_4pct']:.3f} "
              f"| ±8%: {met['auc_camera_8pct']:.3f} | robust AUC {met['robust_auc']:.3f}")

    best = max(results, key=lambda m: results[m]["robust_auc"])
    thr = threshold_for_sensitivity(y, oof_all[best], 0.90)
    screening = binary_metrics(y, oof_all[best], thr)
    print(f"  chosen: {best}; at screening threshold {thr:.3f}: "
          f"sens {screening['sensitivity']:.3f}, spec {screening['specificity']:.3f}")

    Xall, yall = rows(np.arange(n))
    final = candidate_models()[best]
    final.fit(Xall, yall)
    joblib.dump({"model": final, "features": feat_cols, "threshold": thr,
                 "site": site, "model_name": best, "auc": results[best]["auc"],
                 "feat_lo": np.percentile(Xall, 0.5, axis=0), "feat_hi": np.percentile(Xall, 99.5, axis=0),
                 "jitter": [jitter_sd, jitter_copies]},
                os.path.join(out_dir, f"{site}_model.joblib"))
    pd.DataFrame({"rel": d["rel"], "group": groups, "label": y, "oof_prob": oof_all[best]}) \
        .to_csv(os.path.join(out_dir, f"{site}_oof.csv"), index=False)
    return {"n_images": n, "n_unique_photos": n_groups, "best_model": best,
            "cv_default": results[best], "cv_screening": screening, "all_models": results}


def _fresh(model):
    """Untrained copy of a model (works for sklearn and Snap ML estimators)."""
    try:
        from sklearn.base import clone
        return clone(model)
    except Exception:
        import copy
        return copy.deepcopy(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with the image datasets")
    ap.add_argument("--out", default="models")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--jitter-sd", type=float, default=0.08)
    ap.add_argument("--jitter-copies", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    table = build_table(a.data, os.path.join(a.out, "features.csv"), a.workers)
    feat_cols = [c for c in table.columns if c not in META]

    report = {}
    for site, d in table.groupby("site"):
        if site == "unknown" or d["label"].nunique() < 2:
            print(f"\nskipping site '{site}' ({len(d)} images): unknown site or single class")
            continue
        report[site] = train_site(site, d, feat_cols, a.out, a.folds,
                                  a.jitter_sd, a.jitter_copies, a.workers)

    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved models and report.json to {a.out}/")


if __name__ == "__main__":
    main()
