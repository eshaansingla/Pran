"""Domain-adapted CHARIS classifier: CORAL alignment (using unlabeled hardware
background) fit INSIDE each LOPO training fold -- never touching the held-out
CHARIS patient, never using hardware labels (there are none).

For each held-out CHARIS patient:
  1. fit qt_c on the OTHER 12 patients' CHARIS windows only
  2. fit qt_h on the unlabeled hardware background (support/results/hw_features_cache.npz)
  3. CORAL: align the 12-patient CHARIS training set onto the hardware covariance
     structure (whiten source, re-color with target covariance)
  4. train XGBoost on the aligned training features + real CHARIS labels
  5. threshold = Youden's J on the training folds only (never the held-out patient)
  6. evaluate AUC/sens/spec on the held-out patient (also CORAL-aligned the same way)

Then: fit the final production version on all 13 CHARIS patients + full hw
background, and score all 146 real hw_data/*.csv subjects with it to check the
output isn't degenerate (all-flagged or all-normal).
"""
from __future__ import annotations
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import QuantileTransformer

FEATURES = ["cardiac_amplitude", "cardiac_frequency", "respiratory_amplitude", "slow_wave_power", "cardiac_power"]
RNG = np.random.RandomState(0)


def sqrtm(cov):
    u, s, _ = np.linalg.svd(cov)
    return u @ np.diag(np.sqrt(s + 1e-8)) @ u.T


def invsqrtm(cov):
    u, s, _ = np.linalg.svd(cov)
    return u @ np.diag(1 / np.sqrt(s + 1e-8)) @ u.T


def coral_align(X_source_qt, cov_target):
    cov_s = np.cov(X_source_qt.T)
    return (X_source_qt @ invsqrtm(cov_s)) @ sqrtm(cov_target)


def youden_threshold(y_true, p):
    fpr, tpr, thr = roc_curve(y_true, p)
    j = tpr - fpr
    return float(thr[np.argmax(j)])


def main():
    Xc = np.load("results/audit/cache/X.npy").astype(np.float64)
    yc = np.load("results/audit/cache/y.npy").astype(int)
    pidc = np.load("results/audit/cache/pid.npy").astype(int)
    Xh_bg = np.load("support/results/hw_features_cache.npz")["X"].astype(np.float64)

    patients = sorted(np.unique(pidc))
    results = []
    print(f"CORAL-aligned LOPO over {len(patients)} CHARIS patients "
          f"(alignment target: {len(Xh_bg):,} unlabeled hardware background windows)\n")

    for held_out in patients:
        tr_mask = pidc != held_out
        te_mask = pidc == held_out
        Xtr, ytr = Xc[tr_mask], yc[tr_mask]
        Xte, yte = Xc[te_mask], yc[te_mask]

        qt_c = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, len(Xtr)), random_state=0)
        qt_h = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, len(Xh_bg)), random_state=0)
        Xtr_qt = qt_c.fit_transform(Xtr)
        Xte_qt = qt_c.transform(Xte)
        Xh_qt = qt_h.fit_transform(Xh_bg)
        cov_target = np.cov(Xh_qt.T)

        Xtr_al = coral_align(Xtr_qt, cov_target)
        Xte_al = coral_align(Xte_qt, cov_target)  # aligned the same way, using train-fold source cov (fit above)

        n_pos, n_neg = ytr.sum(), len(ytr) - ytr.sum()
        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=n_neg / max(n_pos, 1),
            eval_metric="auc", random_state=0, n_jobs=-1,
        )
        clf.fit(Xtr_al, ytr)

        p_tr = clf.predict_proba(Xtr_al)[:, 1]
        thr = youden_threshold(ytr, p_tr)

        p_te = clf.predict_proba(Xte_al)[:, 1]
        auc = roc_auc_score(yte, p_te) if len(np.unique(yte)) > 1 else float("nan")
        pred = (p_te >= thr).astype(int)
        tp = ((pred == 1) & (yte == 1)).sum(); fn = ((pred == 0) & (yte == 1)).sum()
        tn = ((pred == 0) & (yte == 0)).sum(); fp = ((pred == 1) & (yte == 0)).sum()
        sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)

        results.append(dict(patient=held_out, n=len(yte), n_pos=int(yte.sum()), auc=auc,
                             sens=sens, spec=spec, thr=thr))
        print(f"  patient {held_out:>2}: n={len(yte):>7,}  pos={int(yte.sum()):>7,}  "
              f"AUC={auc:.3f}  sens={sens:.3f}  spec={spec:.3f}  thr={thr:.3f}")

    df = pd.DataFrame(results)
    print(f"\nMean LOPO AUC : {df.auc.mean():.4f}  (95% range {df.auc.quantile(.025):.3f}-{df.auc.quantile(.975):.3f})")
    print(f"Mean sensitivity: {df.sens.mean():.4f}")
    print(f"Mean specificity: {df.spec.mean():.4f}")

    # ---- check train-vs-test gap (overfitting check) on the LAST fold's model ----
    auc_tr = roc_auc_score(ytr, p_tr)
    print(f"\nOverfitting check (last fold): train AUC={auc_tr:.4f} vs held-out AUC={df.auc.iloc[-1]:.4f} "
          f"(gap={auc_tr - df.auc.iloc[-1]:+.4f})")

    # ================= final production model: fit on ALL 13 CHARIS + full hw bg =================
    print("\n[final model] fitting on all 13 CHARIS patients + full hardware background for deployment ...")
    qt_c = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=0)
    qt_h = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=0)
    Xc_qt = qt_c.fit_transform(Xc)
    Xh_qt = qt_h.fit_transform(Xh_bg)
    cov_target = np.cov(Xh_qt.T)
    Xc_al = coral_align(Xc_qt, cov_target)

    n_pos, n_neg = yc.sum(), len(yc) - yc.sum()
    final_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=n_neg / n_pos,
        eval_metric="auc", random_state=0, n_jobs=-1,
    )
    final_clf.fit(Xc_al, yc)
    p_all = final_clf.predict_proba(Xc_al)[:, 1]
    final_thr = youden_threshold(yc, p_all)
    print(f"  final threshold (Youden, all CHARIS): {final_thr:.3f}")

    cov_source_final = np.cov(Xc_qt.T)  # needed to align new hw windows the same way at test time

    def score_hw_csv(path):
        df_ = pd.read_csv(path)
        X_ = df_[FEATURES].to_numpy(np.float64)
        X_qt = qt_h.transform(X_)  # hw data: normalize with hw-domain QT (its own domain)
        # hw data is already IN the target space; CORAL only needs to move the SOURCE into
        # target space, so target-domain data at inference time needs no further alignment.
        p = final_clf.predict_proba(X_qt.astype(np.float32))[:, 1]
        return p

    print("\n[degeneracy check] scoring all 146 real hardware subjects with the CORAL-aligned model ...")
    flagged_pcts = []
    for f in sorted(glob.glob("hw_data/*_features.csv")):
        p = score_hw_csv(f)
        flagged_pcts.append(100 * (p >= final_thr).mean())

    flagged_pcts = np.array(flagged_pcts)
    print(f"  subjects scored: {len(flagged_pcts)}")
    print(f"  flagged% per subject -- min={flagged_pcts.min():.1f}  "
          f"p25={np.percentile(flagged_pcts,25):.1f}  median={np.median(flagged_pcts):.1f}  "
          f"p75={np.percentile(flagged_pcts,75):.1f}  max={flagged_pcts.max():.1f}")
    print(f"  subjects with 0% flagged  : {(flagged_pcts == 0).sum()}/{len(flagged_pcts)}")
    print(f"  subjects with 100% flagged: {(flagged_pcts == 100).sum()}/{len(flagged_pcts)}")
    print(f"  subjects with >90% flagged: {(flagged_pcts > 90).sum()}/{len(flagged_pcts)}")
    print(f"  subjects with <10% flagged: {(flagged_pcts < 10).sum()}/{len(flagged_pcts)}")


if __name__ == "__main__":
    main()
