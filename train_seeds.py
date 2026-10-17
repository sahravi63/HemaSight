"""Multi-seed training and evaluation for HemaSight (run on IBM LinuxONE).

    python3 train_seeds.py --data ~/data --out models_seeds --seeds 5

For every model it repeats training/validation with several random seeds.
A seed changes (a) which colour-jittered copies of each photo are used,
(b) the model's own randomness and (c) for palm/nail, the fold split.

Prints mean +/- std across seeds (how stable the model is), then the score of
the SEED ENSEMBLE (average of all seeds' models), and saves the ensembles as
app-ready models in --out. The current models/ folder is left untouched.

Validation (unchanged from train.py / train_hb.py):
  eyelid Hb  : leave-one-hospital-out (train on 9 hospitals, test on the 10th)
  palm, nail, eyelid yes/no : grouped stratified 5-fold (copies of a photo stay together)
  every model is also scored on simulated new cameras (+/-4 %, +/-8 % colour)
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.metrics import roc_auc_score

from hemasight.models import SeedEnsemble, binary_metrics, stratified_group_folds, threshold_for_sensitivity
from hemasight.robust import features_with_gains, jitter_gains

warnings.filterwarnings("ignore")
CAMERAS = {"camera_4pct": (0.04, 999), "camera_8pct": (0.08, 998)}


def log(msg):
    print(msg, flush=True)


def cached(path, make):
    if os.path.exists(path):
        return np.load(path)
    arr = make()
    np.save(path, arr)
    return arr


def jitter_pool(paths, cols, X0, sd, pool, tag, cache_dir, workers):
    """`pool` jittered feature copies, computed once and cached; seeds sample from it."""
    out = []
    for k in range(pool):
        f = os.path.join(cache_dir, f"jit_{tag}_sd{sd}_{k}_n{len(paths)}.npy")
        X = cached(f, lambda: features_with_gains(paths, jitter_gains(len(paths), sd, 500 + k), cols, workers))
        bad = np.isnan(X).any(axis=1)
        X[bad] = X0[bad]
        out.append(X.astype(np.float32))
    return out


def camera_sets(paths, cols, X0, tag, cache_dir, workers):
    out = {}
    for name, (sd, seed) in CAMERAS.items():
        f = os.path.join(cache_dir, f"cam_{tag}_{name}_n{len(paths)}.npy")
        X = cached(f, lambda: features_with_gains(paths, jitter_gains(len(paths), sd, seed), cols, workers))
        bad = np.isnan(X).any(axis=1)
        X[bad] = X0[bad]
        out[name] = X.astype(np.float32)
    return out


def mean_std(vals):
    v = np.asarray(vals, dtype=float)
    return f"{v.mean():.3f} ± {v.std():.3f}"


# ------------------------------------------------------------------ eyelid Hb
def run_hb(a, out_dir, cache_dir):
    from train_hb import META, load_table
    t = load_table(a.data, os.path.join(a.data, "Anemia_Data_Collection_Sheet.xlsx"),
                   os.path.join(cache_dir, "features_hb.csv"))
    cols = [c for c in t.columns if c not in META]
    hb = t.hb.values.astype(float); y = (hb < 11).astype(int); hosp = t.HOSPITAL.values; n = len(t)
    X0 = t[cols].values.astype(np.float32); paths = t.path.tolist()
    log(f"\n=== eyelid Hb: {n} photos, {len(set(hosp))} hospitals, {a.seeds} seeds ===")
    pool = jitter_pool(paths, cols, X0, a.jitter_sd, a.jitter_pool, "hb", cache_dir, a.workers)
    cams = camera_sets(paths, cols, X0, "hb", cache_dir, a.workers)
    tests = {"normal": X0, **cams}
    hosps = pd.unique(hosp)

    makers = {"extra_trees": lambda s: ExtraTreesRegressor(a.trees, min_samples_leaf=2, max_features=0.5,
                                                          n_jobs=-1, random_state=s),
              "random_forest": lambda s: RandomForestRegressor(a.trees, min_samples_leaf=2, max_features=0.5,
                                                              n_jobs=-1, random_state=s)}
    summary, best_name, best_score, best_preds = {}, None, -1, None
    for name, make in makers.items():
        per_seed, P_all = [], []
        for s in range(a.seeds):
            rng = np.random.default_rng(s)
            pick = rng.choice(len(pool), size=a.jitter_copies, replace=False)
            P = {k: np.zeros(n) for k in tests}
            for h in hosps:
                tr = np.where(hosp != h)[0]; te = hosp == h
                Xtr = np.vstack([X0[tr]] + [pool[k][tr] for k in pick])
                m = make(1000 + s).fit(Xtr, np.tile(hb[tr], 1 + len(pick)))
                for k, X in tests.items():
                    P[k][te] = m.predict(X[te])
            r = {k: (float(np.abs(p - hb).mean()), float(roc_auc_score(y, -p))) for k, p in P.items()}
            per_seed.append(r); P_all.append(P)
            log(f"  {name:13s} seed {s}: normal MAE {r['normal'][0]:.3f} AUC {r['normal'][1]:.3f} | "
                f"±4% AUC {r['camera_4pct'][1]:.3f} | ±8% AUC {r['camera_8pct'][1]:.3f}")
        ens = {k: np.mean([P[k] for P in P_all], axis=0) for k in tests}
        er = {k: (float(np.abs(p - hb).mean()), float(roc_auc_score(y, -p))) for k, p in ens.items()}
        robust = float(np.mean([er[k][1] for k in tests]))
        summary[name] = {
            "per_seed_mean_std": {k: {"mae": mean_std([r[k][0] for r in per_seed]),
                                      "auc": mean_std([r[k][1] for r in per_seed])} for k in tests},
            "seed_ensemble": {k: {"mae": round(er[k][0], 3), "auc": round(er[k][1], 3)} for k in tests},
            "robust_auc": round(robust, 3)}
        log(f"  {name:13s} across seeds: normal AUC {summary[name]['per_seed_mean_std']['normal']['auc']}, "
            f"MAE {summary[name]['per_seed_mean_std']['normal']['mae']}")
        log(f"  {name:13s} SEED ENSEMBLE: normal MAE {er['normal'][0]:.3f} AUC {er['normal'][1]:.3f} | "
            f"±4% {er['camera_4pct'][1]:.3f} | ±8% {er['camera_8pct'][1]:.3f} | robust {robust:.3f}")
        if robust > best_score:
            best_name, best_score, best_preds = name, robust, ens["normal"]

    cut = -threshold_for_sensitivity(y, -best_preds, 0.90)
    scr = binary_metrics(y, -best_preds, -cut)
    log(f"  chosen: {best_name} seed ensemble; flag if Hb < {cut:.2f} -> sens {scr['sensitivity']:.3f}, "
        f"spec {scr['specificity']:.3f}")
    members = []
    for s in range(a.seeds):
        pick = np.random.default_rng(s).choice(len(pool), size=a.jitter_copies, replace=False)
        Xtr = np.vstack([X0] + [pool[k] for k in pick])
        members.append(makers[best_name](1000 + s).fit(Xtr, np.tile(hb, 1 + len(pick))))
    Xall = np.vstack([X0] + pool)
    joblib.dump({"model": SeedEnsemble(members), "features": cols, "uses_demographics": False,
                 "feat_lo": np.percentile(Xall, 0.5, axis=0), "feat_hi": np.percentile(Xall, 99.5, axis=0),
                 "hb_range": (float(hb.min()), float(hb.max())),
                 "auc": summary[best_name]["seed_ensemble"]["normal"]["auc"], "screen_cut": float(cut),
                 "mae": summary[best_name]["seed_ensemble"]["normal"]["mae"],
                 "model_name": f"{best_name} x{a.seeds} seeds", "site": "eye", "kind": "hb_regressor"},
                os.path.join(out_dir, "eye_hb_model.joblib"))
    return {"chosen": best_name, "screening": scr, "models": summary}


# ------------------------------------------------------- palm / nail / eyelid yes-no
def run_classifiers(a, out_dir, cache_dir):
    from train import META, build_table
    table = build_table(a.data, os.path.join(cache_dir, "features.csv"), a.workers)
    cols = [c for c in table.columns if c not in META]
    makers = {"extra_trees": lambda s: ExtraTreesClassifier(a.trees, min_samples_leaf=2, class_weight="balanced",
                                                           n_jobs=-1, random_state=s),
              "random_forest": lambda s: RandomForestClassifier(a.trees, min_samples_leaf=2, class_weight="balanced",
                                                               n_jobs=-1, random_state=s)}
    report = {}
    for site in [s for s in a.sites if s != "eye_hb"]:
        d = table[(table.site == site) & table.label.notna()].reset_index(drop=True)
        if d.empty or d.label.nunique() < 2:
            log(f"\n(skip {site}: no labelled images)")
            continue
        y = d.label.astype(int).values; g = d.group.values; n = len(d)
        X0 = d[cols].values.astype(np.float32); paths = d.path.tolist()
        log(f"\n=== {site}: {n} images, {len(set(g))} unique photos, {a.seeds} seeds ===")
        t0 = time.time()
        pool = jitter_pool(paths, cols, X0, a.jitter_sd, a.jitter_pool, site, cache_dir, a.workers)
        cams = camera_sets(paths, cols, X0, site, cache_dir, a.workers)
        log(f"  colour-jitter and camera features ready ({time.time() - t0:.0f}s)")
        tests = {"normal": X0, **cams}
        summary, best_name, best_score, best_oof = {}, None, -1, None
        for name, make in makers.items():
            per_seed, P_all = [], []
            for s in range(a.seeds):
                pick = np.random.default_rng(s).choice(len(pool), size=a.jitter_copies, replace=False)
                P = {k: np.zeros(n) for k in tests}
                for tr, te in stratified_group_folds(y, g, k=5, seed=s):
                    Xtr = np.vstack([X0[tr]] + [pool[k][tr] for k in pick])
                    m = make(2000 + s).fit(Xtr, np.tile(y[tr], 1 + len(pick)))
                    for k, X in tests.items():
                        P[k][te] = m.predict_proba(X[te])[:, 1]
                r = {k: float(roc_auc_score(y, p)) for k, p in P.items()}
                per_seed.append(r); P_all.append(P)
                log(f"  {name:13s} seed {s}: AUC {r['normal']:.3f} | ±4% {r['camera_4pct']:.3f} | ±8% {r['camera_8pct']:.3f}")
            ens = {k: np.mean([P[k] for P in P_all], axis=0) for k in tests}
            er = {k: float(roc_auc_score(y, p)) for k, p in ens.items()}
            robust = float(np.mean(list(er.values())))
            summary[name] = {"per_seed_mean_std": {k: mean_std([r[k] for r in per_seed]) for k in tests},
                             "seed_ensemble": {k: round(v, 3) for k, v in er.items()}, "robust_auc": round(robust, 3)}
            log(f"  {name:13s} across seeds: AUC {summary[name]['per_seed_mean_std']['normal']}")
            log(f"  {name:13s} SEED ENSEMBLE: AUC {er['normal']:.3f} | ±4% {er['camera_4pct']:.3f} | "
                f"±8% {er['camera_8pct']:.3f} | robust {robust:.3f}")
            if robust > best_score:
                best_name, best_score, best_oof = name, robust, ens["normal"]
        thr = threshold_for_sensitivity(y, best_oof, 0.90)
        scr = binary_metrics(y, best_oof, thr)
        log(f"  chosen: {best_name} seed ensemble; threshold {thr:.3f} -> sens {scr['sensitivity']:.3f}, "
            f"spec {scr['specificity']:.3f}")
        members = []
        for s in range(a.seeds):
            pick = np.random.default_rng(s).choice(len(pool), size=a.jitter_copies, replace=False)
            Xtr = np.vstack([X0] + [pool[k] for k in pick])
            members.append(makers[best_name](2000 + s).fit(Xtr, np.tile(y, 1 + len(pick))))
        Xall = np.vstack([X0] + pool)
        joblib.dump({"model": SeedEnsemble(members), "features": cols, "threshold": float(thr), "site": site,
                     "model_name": f"{best_name} x{a.seeds} seeds", "auc": summary[best_name]["seed_ensemble"]["normal"],
                     "feat_lo": np.percentile(Xall, 0.5, axis=0), "feat_hi": np.percentile(Xall, 99.5, axis=0)},
                    os.path.join(out_dir, f"{site}_model.joblib"))
        report[site] = {"chosen": best_name, "screening": scr, "models": summary}
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models_seeds")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--trees", type=int, default=400, help="trees per model per seed")
    ap.add_argument("--jitter-sd", type=float, default=0.08)
    ap.add_argument("--jitter-copies", type=int, default=4, help="jittered copies used per seed")
    ap.add_argument("--jitter-pool", type=int, default=8, help="jittered copies computed once and sampled")
    ap.add_argument("--sites", default="eye_hb,palm,nail,eye")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args()
    a.sites = [s.strip() for s in a.sites.split(",") if s.strip()]
    os.makedirs(a.out, exist_ok=True)
    cache_dir = os.path.join(a.out, "cache"); os.makedirs(cache_dir, exist_ok=True)
    import platform
    t0 = time.time()
    log(f"machine {platform.machine()}, {os.cpu_count()} CPUs | seeds {a.seeds}, trees {a.trees}, "
        f"jitter sd {a.jitter_sd} x{a.jitter_copies} (pool {a.jitter_pool})")
    report = {"settings": {k: v for k, v in vars(a).items()}, "machine": platform.machine()}
    if "eye_hb" in a.sites:
        report["eye_hb"] = run_hb(a, a.out, cache_dir)
    if any(s != "eye_hb" for s in a.sites):
        report.update(run_classifiers(a, a.out, cache_dir))
    report["minutes"] = round((time.time() - t0) / 60, 1)
    with open(os.path.join(a.out, "report_seeds.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"\nsaved seed-ensemble models and report_seeds.json to {a.out}/ ({report['minutes']} min)")


if __name__ == "__main__":
    main()
