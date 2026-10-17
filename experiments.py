"""Re-run the HemaSight design experiments on this machine (meant for IBM LinuxONE),
so every number in RESULTS.md comes from IBM Z.

    python3 experiments.py --data ~/data --out models [--resnet resnet18-f37072fd.pth]

Eyelid Hb model, leave-one-hospital-out (train on 9 hospitals, test on the 10th):
  1. colour features, plain training
  2. colour features + colour-jitter training            (what the app uses)
  3. pretrained ResNet-18 features (pure numpy, no PyTorch needed on s390x)
Each is also tested on simulated new phone cameras (+/-4 % and +/-8 % colour shift).
Writes models/experiments.json.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from hemasight.robust import features_with_gains, jitter_gains
from train_hb import META, load_table

warnings.filterwarnings("ignore")


def _cnn_one(args):
    path, weights = args
    global _CNN
    try:
        _CNN
    except NameError:
        from hemasight.cnn import ResNet18Features
        _CNN = ResNet18Features(weights)
    return _CNN(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models")
    ap.add_argument("--resnet", default=None, help="path to resnet18-f37072fd.pth (optional)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    print(f"machine: {platform.machine()} ({platform.platform()}), {os.cpu_count()} CPUs")

    t = load_table(a.data, os.path.join(a.data, "Anemia_Data_Collection_Sheet.xlsx"),
                   os.path.join(a.out, "features_hb.csv"))
    cols = [c for c in t.columns if c not in META]
    hb = t.hb.values.astype(float); y = (hb < 11).astype(int); hosp = t.HOSPITAL.values; n = len(t)
    X0 = t[cols].values.astype(np.float32); paths = t.path.tolist()
    cams = {"camera_4pct": features_with_gains(paths, jitter_gains(n, 0.04, 999), cols),
            "camera_8pct": features_with_gains(paths, jitter_gains(n, 0.08, 998), cols)}
    jit = [features_with_gains(paths, jitter_gains(n, 0.08, 100 + k), cols) for k in range(4)]
    for X in list(cams.values()) + jit:
        bad = np.isnan(X).any(axis=1); X[bad] = X0[bad]

    def et():
        return ExtraTreesRegressor(500, min_samples_leaf=2, max_features=0.5, n_jobs=-1, random_state=42)

    def loho(fit_rows, tests, make):
        P = {k: np.zeros(n) for k in tests}
        for h in pd.unique(hosp):
            te = hosp == h
            m = make().fit(*fit_rows(np.where(~te)[0]))
            for k, X in tests.items():
                P[k][te] = m.predict(X[te])
        return {k: {"mae": round(float(np.abs(p - hb).mean()), 3),
                    "auc": round(float(roc_auc_score(y, -p)), 3)} for k, p in P.items()}

    tests = {"normal": X0, **cams}
    res = {}
    res["colour_plain"] = loho(lambda i: (X0[i], hb[i]), tests, et)
    res["colour_jitter_training"] = loho(
        lambda i: (np.vstack([X0[i]] + [J[i] for J in jit]), np.tile(hb[i], 1 + len(jit))), tests, et)

    if a.resnet and os.path.exists(a.resnet):
        from concurrent.futures import ProcessPoolExecutor
        t1 = time.time()
        with ProcessPoolExecutor() as ex:
            C = np.stack(list(ex.map(_cnn_one, [(p, a.resnet) for p in paths], chunksize=4)))
        print(f"ResNet-18 features (numpy on {platform.machine()}): {time.time() - t1:.0f}s for {n} photos")
        svr = lambda: make_pipeline(StandardScaler(), SVR(C=3, epsilon=0.3))
        res["resnet18_features"] = loho(lambda i: (C[i], hb[i]), {"normal": C}, svr)

    base = np.zeros(n)
    for h in pd.unique(hosp):
        base[hosp == h] = hb[hosp != h].mean()
    res["baseline_predict_average"] = {"normal": {"mae": round(float(np.abs(base - hb).mean()), 3), "auc": 0.5}}

    print("\nEyelid Hb model, unseen hospitals:")
    for k, r in res.items():
        line = " | ".join(f"{c}: MAE {v['mae']:.2f} AUC {v['auc']:.3f}" for c, v in r.items())
        print(f"  {k:26s} {line}")
    with open(os.path.join(a.out, "experiments.json"), "w") as f:
        json.dump({"machine": platform.machine(), "platform": platform.platform(),
                   "cpus": os.cpu_count(), "minutes": round((time.time() - t0) / 60, 1), "results": res}, f, indent=2)
    print(f"\nsaved {a.out}/experiments.json ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
