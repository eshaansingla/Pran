"""Domain-blind test for a new sensor CSV.

1. Can a classifier tell "CHARIS window" vs "hardware window" apart from the raw
   5-feature vectors alone? (expect yes -- absolute scales differ by ~1000x)
2. Fix it with INDEPENDENT per-domain scaling (QuantileTransformer fit separately
   on CHARIS and on hardware background), same idea as hybrid_pipeline_v4's
   domain-separated QT.
3. Re-run the domain classifier on the scaled features -- confirm it can no longer
   tell the domains apart (AUC -> ~0.5).
4. Score the new sensor CSV with the real CHARIS XGBoost model using the
   domain-appropriate scaling, report results.

Usage: python charis/domain_blind_test.py <path_to_sensor_csv>
"""
from __future__ import annotations
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import QuantileTransformer

ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES = ["cardiac_amplitude", "cardiac_frequency", "respiratory_amplitude", "slow_wave_power", "cardiac_power"]
CHARIS_CACHE = ROOT / "results" / "audit" / "cache"
HW_CACHE = ROOT / "support" / "results" / "hw_features_cache.npz"
MODEL_DIR = ROOT / "models" / "charis_compare" / "XGBoost"
RNG = np.random.RandomState(0)


def domain_auc(Xc: np.ndarray, Xh: np.ndarray, label: str):
    X = np.vstack([Xc, Xh])
    y = np.concatenate([np.zeros(len(Xc)), np.ones(len(Xh))])
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0, n_jobs=-1)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    aucs = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    print(f"  [{label}] domain-classifier AUC (CHARIS vs hardware): {aucs.mean():.4f} +/- {aucs.std():.4f}")
    return aucs.mean()


def main(sensor_csv: Path):
    # ---- load domains -------------------------------------------------------
    Xc_full = np.load(CHARIS_CACHE / "X.npy").astype(np.float64)
    if len(Xc_full) > 20000:
        Xc_full = Xc_full[RNG.choice(len(Xc_full), 20000, replace=False)]

    hw = np.load(HW_CACHE)
    Xh_bg = hw["X"].astype(np.float64)  # 146-subject background pool

    df = pd.read_csv(sensor_csv)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        sys.exit(f"CSV is missing columns: {missing}")
    Xh_test = df[FEATURES].to_numpy(np.float64)

    print(f"CHARIS background : {len(Xc_full):,} windows (sampled from {915137:,})")
    print(f"Hardware background: {len(Xh_bg):,} windows (146 subjects)")
    print(f"New sensor CSV     : {len(Xh_test):,} windows ({sensor_csv.name})\n")

    # ---- step 1: can it tell domains apart, raw scale? -----------------------
    print("[1] RAW features (no scaling) -- domain classifier")
    domain_auc(Xc_full, Xh_bg, "raw")
    print(f"    raw feature means -- CHARIS: {Xc_full.mean(0).round(3)}")
    print(f"    raw feature means -- hardware: {Xh_bg.mean(0).round(3)}\n")

    # ---- step 2: independent per-domain scaling -------------------------------
    print("[2] Fitting INDEPENDENT QuantileTransformers per domain (CHARIS-only, hardware-only)")
    qt_c = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, len(Xc_full)), random_state=0)
    qt_h = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, len(Xh_bg)), random_state=0)
    Xc_qt = qt_c.fit_transform(Xc_full)
    Xh_bg_qt = qt_h.fit_transform(Xh_bg)
    Xh_test_qt = qt_h.transform(Xh_test)  # new CSV scaled with the SAME hardware-domain transform
    print("    each domain now independently mapped to its own N(0,1) quantile space\n")

    # ---- step 3: re-test domain classifier on scaled features -----------------
    print("[3] SCALED features -- domain classifier (should now be near-blind)")
    domain_auc(Xc_qt, Xh_bg_qt, "domain-blind QT (marginals matched)")
    print("    -> matching marginals alone is not enough; a nonlinear classifier can still\n"
          "       exploit differences in the CORRELATION structure between the 5 features.\n"
          "       Whitening each domain (decorrelate + unit variance) on top of the QT step:")

    def whiten(X, ref=None):
        mu = X.mean(0) if ref is None else ref[0]
        cov = np.cov((X - mu).T) if ref is None else ref[1]
        # symmetric (ZCA) whitening -- keeps feature identity/interpretability, unlike PCA whitening
        u, s, _ = np.linalg.svd(cov)
        Wm = u @ np.diag(1.0 / np.sqrt(s + 1e-8)) @ u.T
        return (X - mu) @ Wm, (mu, cov, Wm)

    Xc_w, ref_c = whiten(Xc_qt)
    Xh_bg_w, ref_h = whiten(Xh_bg_qt)
    Xh_test_w = (Xh_test_qt - ref_h[0]) @ ref_h[2]
    domain_auc(Xc_w, Xh_bg_w, "domain-blind QT + whitened")
    print()

    # ---- step 4: score the sensor CSV with the real CHARIS model --------------
    print("[4] Scoring the new sensor CSV with the trained CHARIS XGBoost model")
    if not (MODEL_DIR / "model.pkl").exists():
        sys.exit(f"model not found at {MODEL_DIR}")
    model = pickle.load(open(MODEL_DIR / "model.pkl", "rb"))
    import json
    info = json.loads((MODEL_DIR / "metrics.json").read_text())
    threshold = float(info["threshold"])

    # (a) using the model's OWN CHARIS-fit scaler (native pipeline, not domain-blind)
    qt_native = pickle.load(open(MODEL_DIR / "qt_scaler.pkl", "rb"))
    p_native = model.predict_proba(qt_native.transform(Xh_test).astype(np.float32))[:, 1]

    # (b) using the domain-blind hardware-domain QT from step 2 (marginals matched only)
    p_blind = model.predict_proba(Xh_test_qt.astype(np.float32))[:, 1]
    # (c) using the fully domain-blind QT+whitened features from step 3
    p_blind_w = model.predict_proba(Xh_test_w.astype(np.float32))[:, 1]

    for name, p in [("native CHARIS scaler (as-deployed)", p_native),
                     ("domain-blind hardware-QT, marginals only", p_blind),
                     ("domain-blind hardware-QT + whitened (fully domain-blind)", p_blind_w)]:
        flagged = (p >= threshold).mean() * 100
        print(f"  {name}:")
        print(f"    mean P(elevated) = {p.mean():.4f}  |  windows flagged >= threshold({threshold:.3f}) = {flagged:.1f}%")

    out = df.copy()
    out["p_elevated_native"] = np.round(p_native, 4)
    out["p_elevated_domain_blind_marginals"] = np.round(p_blind, 4)
    out["p_elevated_domain_blind_whitened"] = np.round(p_blind_w, 4)
    out_path = sensor_csv.with_name(sensor_csv.stem + "_scored.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved per-window scores -> {out_path}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\Users\asus\Downloads\tes"))
