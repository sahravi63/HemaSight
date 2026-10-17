"""Model zoo and cross-validation helpers that work across scikit-learn
versions (Ubuntu's packaged sklearn on s390x can be old)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import warnings

try:  # sklearn < 1.0 needs the experimental import; newer versions just warn
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
except Exception:
    pass
# Snap ML prints a warning per model when the IBM Z AI accelerator (NNPA) is not exposed
warnings.filterwarnings("ignore", message=".*NNPA.*")
from sklearn.ensemble import HistGradientBoostingClassifier


def candidate_models(seed: int = 42) -> dict:
    models = {
        "logreg": make_pipeline(StandardScaler(),
                                LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced")),
        "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                class_weight="balanced", n_jobs=-1, random_state=seed),
        "extra_trees": ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2,
                                            class_weight="balanced", n_jobs=-1, random_state=seed),
        "hist_gb": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                  max_leaf_nodes=15, l2_regularization=1.0,
                                                  random_state=seed),
    }
    # IBM Snap ML: accelerated trees with native s390x builds, a natural fit on LinuxONE
    try:
        from snapml import BoostingMachineClassifier, SnapRandomForestClassifier
        models["snap_boosting"] = BoostingMachineClassifier(num_round=300, learning_rate=0.05,
                                                            max_depth=4, random_state=seed)
        models["snap_forest"] = SnapRandomForestClassifier(n_estimators=400, random_state=seed,
                                                           n_jobs=4)
    except Exception:
        pass
    return models


def stratified_group_folds(y: np.ndarray, groups: np.ndarray, k: int = 5, seed: int = 42):
    """Yield (train_idx, test_idx). Every group lands in exactly one test fold,
    and folds keep roughly the same anaemic/non-anaemic balance."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    g_label = {g: int(round(y[groups == g].mean())) for g in uniq}
    folds = [[] for _ in range(k)]
    for lab in (0, 1):
        gs = [g for g in uniq if g_label[g] == lab]
        rng.shuffle(gs)
        for i, g in enumerate(gs):
            folds[i % k].append(g)
    for f in folds:
        test = np.isin(groups, f)
        yield np.nonzero(~test)[0], np.nonzero(test)[0]


def binary_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    from sklearn.metrics import roc_auc_score
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-9)
    return {"threshold": round(float(thr), 4), "auc": round(float(roc_auc_score(y, p)), 4),
            "accuracy": round((tp + tn) / len(y), 4), "sensitivity": round(sens, 4),
            "specificity": round(spec, 4), "precision": round(prec, 4), "f1": round(f1, 4),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray, target: float = 0.90) -> float:
    """Highest threshold that still catches at least `target` of anaemic cases."""
    pos = np.sort(p[y == 1])
    if len(pos) == 0:
        return 0.5
    k = int(np.floor((1 - target) * len(pos)))
    return float(pos[k])


def candidate_regressors(seed: int = 42) -> dict:
    from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                                  RandomForestRegressor)
    from sklearn.linear_model import Ridge
    models = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "random_forest": RandomForestRegressor(n_estimators=500, min_samples_leaf=3,
                                               n_jobs=-1, random_state=seed),
        "extra_trees": ExtraTreesRegressor(n_estimators=600, min_samples_leaf=3,
                                           n_jobs=-1, random_state=seed),
        "hist_gb": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04,
                                                 max_leaf_nodes=15, l2_regularization=1.0,
                                                 random_state=seed),
    }
    try:
        from snapml import BoostingMachineRegressor, SnapRandomForestRegressor
        models["snap_boosting"] = BoostingMachineRegressor(num_round=300, learning_rate=0.04,
                                                           max_depth=4, random_state=seed)
        models["snap_forest"] = SnapRandomForestRegressor(n_estimators=500, random_state=seed,
                                                          n_jobs=4)
    except Exception:
        pass
    return models


class SeedEnsemble:
    """Average of the same model trained with different random seeds.

    Each member sees different colour-jitter copies and different internal
    randomness; averaging them gives steadier predictions than any one seed."""

    def __init__(self, models):
        self.models = list(models)

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)
