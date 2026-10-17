"""Train the HemaSight haemoglobin (g/dL) estimator from eyelid photos.

Uses the CP-AnemiC photos (Image_001 ... Image_710) and the lab Hb values in
Anemia_Data_Collection_Sheet.xlsx.

Usage:
    python3 train_hb.py --data ~/data --out models

How the model is made to generalise
  * colour-jitter training: every photo is also seen with random per-channel
    colour gains, so a different phone camera does not throw the model off
  * validation on unseen hospitals: train on 9 hospitals, test on the 10th
  * robustness test: the unseen-hospital photos are also re-coloured as if
    taken by a new camera (+/-4 % and +/-8 %); the model is chosen by its
    average AUC across normal and shifted conditions, not by its best case
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

from hemasight.features import extract_features
from hemasight.models import binary_metrics, stratified_group_folds, threshold_for_sensitivity
from hemasight.robust import features_with_gains, jitter_gains

ANAEMIA_CUTOFF = 11.0  # g/dL, WHO, children 6-59 months
META = {"path", "id", "IMAGE_ID", "HB_LEVEL", "Severity", "Age(Months)", "GENDER", "REMARK",
        "HOSPITAL", "CITY/TOWN", "MUNICIPALITY/DISTRICT", "REGION", "COUNTRY", "hb",
        "age_months", "male", "remark_anaemic", "folder_anaemic"}


def load_table(data: str, sheet: str, cache: str) -> pd.DataFrame:
    s = pd.read_excel(sheet)
    s["id"] = s["IMAGE_ID"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    s["hb"] = pd.to_numeric(s["HB_LEVEL"], errors="coerce")
    s["age_months"] = pd.to_numeric(s["Age(Months)"], errors="coerce")
    s["male"] = (s["GENDER"].astype(str).str.strip().str.lower() == "male").astype(int)
    s["remark_anaemic"] = (~s["REMARK"].astype(str).str.lower().str.startswith("non")).astype(int)

    files = [f for f in glob.glob(os.path.join(data, "**", "*.*"), recursive=True)
             if re.fullmatch(r"image_\d+", os.path.splitext(os.path.basename(f))[0].lower())]
    imgs = pd.DataFrame({"path": files})
    imgs["id"] = imgs["path"].map(lambda p: int(re.search(r"\d+", os.path.basename(p)).group()))
    imgs["folder_anaemic"] = imgs["path"].map(
        lambda p: 0 if os.path.basename(os.path.dirname(p)).lower().startswith("non") else 1)
    t = imgs.merge(s, on="id", how="inner").dropna(subset=["hb"]).sort_values("id").reset_index(drop=True)
    print(f"{len(t)} eyelid photos matched to lab Hb (of {len(s)} sheet rows, {len(imgs)} images), "
          f"{t.HOSPITAL.nunique()} hospitals")
    print(f"Hb range {t.hb.min():.1f}-{t.hb.max():.1f} g/dL, mean {t.hb.mean():.2f}; "
          f"anaemic (Hb<{ANAEMIA_CUTOFF:g}): {int((t.hb < ANAEMIA_CUTOFF).sum())}")
    print(f"folder label vs sheet remark disagreements: {int((t.folder_anaemic != t.remark_anaemic).sum())}")

    if os.path.exists(cache):
        c = pd.read_csv(cache)
        if set(t["path"]) <= set(c["path"]):
            return t.merge(c, on="path")
    with ProcessPoolExecutor() as ex:
        F = pd.DataFrame(list(ex.map(extract_features, t["path"], chunksize=8)))
    F.insert(0, "path", t["path"].values)
    F.to_csv(cache, index=False)
    return t.merge(F, on="path")


def regressors(seed=42):
    m = {"extra_trees": ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.5,
                                            n_jobs=-1, random_state=seed),
         "hist_gb": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15,
                                                  l2_regularization=1.0, random_state=seed)}
    try:
        from snapml import SnapRandomForestRegressor
        m["snap_forest"] = SnapRandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=4)
    except Exception:
        pass
    return m


def summarise(pred, hb, y):
    err = pred - hb
    return {"mae": round(float(np.abs(err).mean()), 3),
            "rmse": round(float(np.sqrt((err ** 2).mean())), 3),
            "pearson_r": round(float(np.corrcoef(pred, hb)[0, 1]), 3),
            "within_1gdl_pct": round(float((np.abs(err) <= 1.0).mean() * 100), 1),
            "auc_anaemia": binary_metrics(y, -pred, float(np.median(-pred)))["auc"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sheet", default=None, help="defaults to <data>/Anemia_Data_Collection_Sheet.xlsx")
    ap.add_argument("--out", default="models")
    ap.add_argument("--jitter-sd", type=float, default=0.08, help="colour-gain jitter for training")
    ap.add_argument("--jitter-copies", type=int, default=4, help="jittered copies per photo")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sheet = a.sheet or os.path.join(a.data, "Anemia_Data_Collection_Sheet.xlsx")

    t = load_table(a.data, sheet, os.path.join(a.out, "features_hb.csv"))
    cols = [c for c in t.columns if c not in META]
    hb = t["hb"].values.astype(float)
    y = (hb < ANAEMIA_CUTOFF).astype(int)
    hosp = t["HOSPITAL"].values
    n = len(t)
    X0 = t[cols].values.astype(np.float32)

    print(f"\ncomputing {a.jitter_copies} colour-jittered copies (sd {a.jitter_sd}) "
          f"and 2 simulated new cameras ...")
    paths = t["path"].tolist()
    Xjit = [features_with_gains(paths, jitter_gains(n, a.jitter_sd, 100 + k), cols)
            for k in range(a.jitter_copies)]
    Xcam = {"camera_4pct": features_with_gains(paths, jitter_gains(n, 0.04, 999), cols),
            "camera_8pct": features_with_gains(paths, jitter_gains(n, 0.08, 998), cols)}
    for X in Xjit + list(Xcam.values()):
        bad = np.isnan(X).any(axis=1)
        X[bad] = X0[bad]

    def train_rows(idx):
        Xtr = np.vstack([X0[idx]] + [Xj[idx] for Xj in Xjit])
        return Xtr, np.tile(hb[idx], 1 + len(Xjit))

    base = np.zeros(n)
    for h in pd.unique(hosp):
        base[hosp == h] = hb[hosp != h].mean()
    print(f"baseline (predict the average Hb), unseen hospitals: MAE {np.abs(base - hb).mean():.3f} g/dL")

    print("\nvalidation on UNSEEN HOSPITALS (train on 9, test on the 10th):")
    results, preds = {}, {}
    for name, model in regressors().items():
        P = {"normal": np.zeros(n), **{k: np.zeros(n) for k in Xcam}}
        try:
            for h in pd.unique(hosp):
                te = hosp == h
                m = clone(model).fit(*train_rows(np.where(~te)[0]))
                P["normal"][te] = m.predict(X0[te])
                for k, Xc in Xcam.items():
                    P[k][te] = m.predict(Xc[te])
        except Exception as e:
            print(f"  {name:12s} failed: {e}")
            continue
        r = {k: summarise(p, hb, y) for k, p in P.items()}
        r["robust_auc"] = round(float(np.mean([r[k]["auc_anaemia"] for k in P])), 3)
        results[name], preds[name] = r, P["normal"]
        print(f"  {name:12s} normal: MAE {r['normal']['mae']:.2f} AUC {r['normal']['auc_anaemia']:.3f} | "
              f"new camera ±4%: AUC {r['camera_4pct']['auc_anaemia']:.3f} | "
              f"±8%: AUC {r['camera_8pct']['auc_anaemia']:.3f} | robust AUC {r['robust_auc']:.3f}")

    best = max(results, key=lambda k: results[k]["robust_auc"])
    p = preds[best]
    cut = -threshold_for_sensitivity(y, -p, 0.90)
    screening = binary_metrics(y, -p, -cut)
    screening["threshold"] = round(float(cut), 3)
    print(f"\nchosen: {best} (best average AUC across normal and new-camera conditions)")
    print(f"unseen hospitals: MAE {results[best]['normal']['mae']:.2f} g/dL, "
          f"within ±1 g/dL {results[best]['normal']['within_1gdl_pct']:.0f}%")
    print(f"screening: flag if estimated Hb < {cut:.2f} g/dL -> "
          f"sens {screening['sensitivity']:.3f}, spec {screening['specificity']:.3f}")

    # also report standard 5-fold CV for comparison with other work
    pf = np.zeros(n)
    for tr, te in stratified_group_folds(y, t["id"].values, k=5):
        pf[te] = clone(regressors()[best]).fit(*train_rows(tr)).predict(X0[te])
    cv5 = summarise(pf, hb, y)
    print(f"5-fold CV (mixed hospitals): MAE {cv5['mae']:.2f}, AUC {cv5['auc_anaemia']:.3f}")

    Xall, yall = train_rows(np.arange(n))
    final = clone(regressors()[best]).fit(Xall, yall)
    joblib.dump({"model": final, "features": cols, "uses_demographics": False,
                 "feat_lo": np.percentile(Xall, 0.5, axis=0), "feat_hi": np.percentile(Xall, 99.5, axis=0),
                 "hb_range": (float(hb.min()), float(hb.max())),
                 "auc": results[best]["normal"]["auc_anaemia"], "screen_cut": float(cut),
                 "mae": results[best]["normal"]["mae"], "model_name": best,
                 "site": "eye", "kind": "hb_regressor", "jitter": [a.jitter_sd, a.jitter_copies]},
                os.path.join(a.out, "eye_hb_model.joblib"))
    pd.DataFrame({"image": t["IMAGE_ID"], "hospital": hosp, "hb_lab": hb,
                  "hb_estimate_unseen_hospital": np.round(p, 2)}) \
        .to_csv(os.path.join(a.out, "eye_hb_oof.csv"), index=False)
    with open(os.path.join(a.out, "report_hb.json"), "w") as f:
        json.dump({"n_photos": n, "n_hospitals": int(t.HOSPITAL.nunique()), "chosen": best,
                   "validation": "leave-one-hospital-out", "screening": screening,
                   "baseline_mae": round(float(np.abs(base - hb).mean()), 3),
                   "cv5_mixed_hospitals": cv5, "jitter": {"sd": a.jitter_sd, "copies": a.jitter_copies},
                   "models": results}, f, indent=2)
    print(f"saved {a.out}/eye_hb_model.joblib and report_hb.json")


if __name__ == "__main__":
    main()
