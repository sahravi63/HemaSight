"""Load trained models and turn photos into a screening decision."""
from __future__ import annotations

import glob
import os

import joblib
import numpy as np

from PIL import Image

from .features import extract_features

SITE_NAMES = {"eye": "Inner eyelid", "nail": "Fingernails", "palm": "Palm"}
UNCERTAIN_BAND = 0.10  # margin around a site's threshold treated as "unsure"
HB_MARGIN_SCALE = 4.0  # g/dL that map to a full probability-sized vote
# training photos for these sites are hand-cut regions on a transparent background,
# so an un-outlined photo (eye, lashes, skin, background) is out of distribution
# (palm training photos are opaque square patches of palm skin, so palm needs no cut-out)
NEEDS_CUTOUT = {"eye", "nail"}
# validated AUC per site (overridden by the value stored at training time)
DEFAULT_AUC = {"eye": 0.93, "palm": 0.87, "nail": 0.75}
SUPPORTING_AUC = 0.80  # sites below this can support a result but never decide it alone


def site_weight(auc: float) -> float:
    """Vote weight from validated AUC: 0 for a coin flip, 1 for a perfect model."""
    return max(0.0, 2 * (auc - 0.5)) ** 2


def is_cutout(path: str, min_transparent: float = 0.03) -> bool:
    """True if the image carries an alpha mask, i.e. the region was outlined."""
    im = Image.open(path)
    if im.mode not in ("RGBA", "LA", "PA") and "transparency" not in im.info:
        return False
    a = np.asarray(im.convert("RGBA"))[..., 3]
    return float((a < 128).mean()) >= min_transparent


def who_grade_child(hb: float) -> str:
    """WHO anaemia grade for children 6-59 months (the population the Hb model was trained on)."""
    if hb < 7.0:
        return "severe"
    if hb < 10.0:
        return "moderate"
    if hb < 11.0:
        return "mild"
    return "none"


OOD_LIMIT = 0.25  # share of features outside the training range that marks a photo as unusual


def ood_fraction(bundle: dict, x: np.ndarray) -> float:
    """Share of features outside the central 99 % of the training photos."""
    lo, hi = bundle.get("feat_lo"), bundle.get("feat_hi")
    if lo is None:
        return 0.0
    return float(((x[0] < lo) | (x[0] > hi)).mean())


class Screener:
    def __init__(self, model_dir: str = "models"):
        self.models, self.hb = {}, None
        for f in glob.glob(os.path.join(model_dir, "*_model.joblib")):
            b = joblib.load(f)
            if b.get("kind") == "hb_regressor":
                self.hb = b
            else:
                self.models[b["site"]] = b
        if not self.models and not self.hb:
            raise FileNotFoundError(f"no models in {model_dir}; run train.py / train_hb.py first")

    @property
    def sites(self):
        s = set(self.models)
        if self.hb:
            s.add("eye")
        return sorted(s, key=["eye", "nail", "palm"].index)

    def _x(self, feats: dict, cols: list) -> np.ndarray:
        return np.array([[feats.get(c, 0.0) for c in cols]], dtype=np.float32)

    def estimate_hb(self, image_path: str, age_months: float = 24, male: int = 0) -> dict:
        f = extract_features(image_path)
        f["age_months"], f["male"] = float(age_months), int(male)
        x = self._x(f, self.hb["features"])
        hb = float(self.hb["model"].predict(x)[0])
        lo, hi = self.hb.get("hb_range", (3.0, 18.0))
        hb = round(float(np.clip(hb, lo, hi)), 1)  # grade the same number the user sees
        cut = self.hb["screen_cut"]
        margin = float(np.clip((cut - hb) / HB_MARGIN_SCALE, -0.5, 0.5))
        return {"site": "eye", "name": SITE_NAMES["eye"], "hb": round(hb, 1),
                "hb_mae": self.hb["mae"], "grade": who_grade_child(hb), "screen_cut": cut,
                "margin": margin, "model": self.hb["model_name"], "ood": ood_fraction(self.hb, x),
                "auc": float(self.hb.get("auc", DEFAULT_AUC["eye"]))}

    def site_probability(self, site: str, image_path: str) -> dict:
        b = self.models[site]
        x = self._x(extract_features(image_path), b["features"])
        p = float(b["model"].predict_proba(x)[0, 1])
        return {"site": site, "name": SITE_NAMES.get(site, site), "probability": p,
                "threshold": float(b["threshold"]), "margin": p - float(b["threshold"]),
                "model": b["model_name"], "ood": ood_fraction(b, x),
                "auc": float(b.get("auc", DEFAULT_AUC.get(site, 0.75)))}

    def screen(self, images: dict, age_months: float = 24, male: int = 0) -> dict:
        """images: {site: path}. Returns per-site results and one combined decision."""
        per_site, hb, skipped = [], None, []
        for site, path in images.items():
            if not path:
                continue
            if site in NEEDS_CUTOUT and not is_cutout(path):
                skipped.append({"site": site, "name": SITE_NAMES.get(site, site),
                                "skipped": "not outlined; outline the region and screen again"})
                continue
            if site == "eye" and self.hb:
                hb = self.estimate_hb(path, age_months, male)
                per_site.append(hb)
            elif site in self.models:
                per_site.append(self.site_probability(site, path))
        for r in per_site:
            r["unusual"] = r.get("ood", 0) > OOD_LIMIT
            r["supporting"] = r["auc"] < SUPPORTING_AUC
            r["weight"] = round(site_weight(r["auc"]), 3)
        if not per_site:
            return {"decision": "needs_outline", "hb": None, "per_site": skipped, "margin": 0.0,
                    "action": ("These photos were not outlined. Click around the inner red eyelid "
                               "lining, the nail, or a patch of palm skin to cut it out, then press "
                               "Screen again. The models were trained on cut-out regions, so a whole "
                               "eye or hand photo gives no reliable reading."),
                    "sites_used": 0, "sites_disagree": False}

        # unusual photos do not vote; the rest vote by how far they sit beyond
        # their own screening threshold, weighted by how accurate that site is
        voters = [r for r in per_site if not r["unusual"]]
        primary = [r for r in voters if not r["supporting"]]
        if voters:
            w = np.array([r["weight"] for r in voters]); m = np.array([r["margin"] for r in voters])
            margin = float((w * m).sum() / max(w.sum(), 1e-9))
        else:
            margin = 0.0
        pm = np.array([r["margin"] for r in primary])
        disagree = bool(len(pm) and (pm > 0).any() and (pm < 0).any())

        if not voters:
            level, action = "uncertain", ("These photos look unlike the training photos (lighting, "
                                          "colour, framing or not a real photo), so there is no reliable "
                                          "reading. Retake in daylight, outline only the tissue, or refer "
                                          "for a blood test.")
        elif hb and not hb["unusual"] and hb["grade"] == "severe":
            level, action = "likely_anaemic", ("Possible severe anaemia. Refer urgently for a "
                                               "confirmatory blood test today.")
        elif not primary:
            level, action = "uncertain", ("Fingernails alone are not reliable enough to decide. "
                                          "Add an inner-eyelid or palm photo, or refer for a blood test.")
        elif abs(margin) < UNCERTAIN_BAND or disagree:
            level, action = "uncertain", ("Result is uncertain. Retake the photos in good light; "
                                          "if still uncertain, refer for a confirmatory blood test.")
        elif margin > 0:
            level, action = "likely_anaemic", ("Signs of anaemia. Refer for a confirmatory blood "
                                               "test, start the follow-up plan and re-screen in 4 weeks.")
        else:
            level, action = "unlikely", ("No sign of anaemia. Give iron-rich diet advice and "
                                         "re-screen at the next routine visit.")
        if skipped:
            action += " (Some photos were skipped because they were not outlined.)"
        return {"decision": level, "action": action, "margin": round(margin, 3), "hb": hb,
                "sites_used": len(per_site), "sites_disagree": disagree,
                "per_site": per_site + skipped}
