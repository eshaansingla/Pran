"""Stricter re-check on top of coral_lopo.py: does the CORAL-aligned model need
each patient's own BASELINE LEVEL to score well, or does it work on purely
within-patient RELATIVE deviation (the only kind of signal that can't leak
between-patient information)?

For each LOPO fold: subtract the held-out patient's own per-feature mean from
their windows (label-blind -- uses only their features, not their labels) before
scoring. If AUC survives de-meaning, the model is reading genuine within-patient
rise/fall, not "which patient is this typically like."
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer

from coral_lopo import coral_align, youden_threshold  # reuse the exact same fitted pipeline logic


def main():
    Xc = np.load("results/audit/cache/X.npy").astype(np.float64)
    yc = np.load("results/audit/cache/y.npy").astype(int)
    pidc = np.load("results/audit/cache/pid.npy").astype(int)
    Xh_bg = np.load("support/results/hw_features_cache.npz")["X"].astype(np.float64)

    patients = sorted(np.unique(pidc))
    rows = []
    print("Patient | AUC (baseline intact) | AUC (patient de-meaned, no baseline info)\n")

    for held_out in patients:
        tr_mask = pidc != held_out
        te_mask = pidc == held_out
        Xtr, ytr = Xc[tr_mask], yc[tr_mask]
        Xte, yte = Xc[te_mask], yc[te_mask]
        if len(np.unique(yte)) < 2:
            print(f"  patient {held_out}: only one class present, AUC undefined, skipped")
            continue

        qt_c = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, len(Xtr)), random_state=0)
        qt_h = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, len(Xh_bg)), random_state=0)
        Xtr_qt = qt_c.fit_transform(Xtr)
        Xte_qt = qt_c.transform(Xte)
        Xh_qt = qt_h.fit_transform(Xh_bg)
        cov_target = np.cov(Xh_qt.T)

        Xtr_al = coral_align(Xtr_qt, cov_target)
        Xte_al = coral_align(Xte_qt, cov_target)

        n_pos, n_neg = ytr.sum(), len(ytr) - ytr.sum()
        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=n_neg / max(n_pos, 1),
            eval_metric="auc", random_state=0, n_jobs=-1,
        )
        clf.fit(Xtr_al, ytr)

        p_intact = clf.predict_proba(Xte_al)[:, 1]
        auc_intact = roc_auc_score(yte, p_intact)

        # de-mean the held-out patient's ALIGNED features by their OWN mean (label-blind)
        Xte_al_demeaned = Xte_al - Xte_al.mean(0, keepdims=True)
        # re-center around the population's mean so the classifier's learned decision
        # boundary (fit around population-centered features) still applies
        Xte_al_demeaned = Xte_al_demeaned + Xtr_al.mean(0, keepdims=True)
        p_demeaned = clf.predict_proba(Xte_al_demeaned)[:, 1]
        auc_demeaned = roc_auc_score(yte, p_demeaned)

        rows.append(dict(patient=held_out, auc_intact=auc_intact, auc_demeaned=auc_demeaned,
                          drop=auc_intact - auc_demeaned))
        print(f"  patient {held_out:>2}: intact={auc_intact:.3f}   de-meaned={auc_demeaned:.3f}   "
              f"drop={auc_intact - auc_demeaned:+.3f}")

    df = pd.DataFrame(rows)
    print(f"\nMean AUC intact    : {df.auc_intact.mean():.4f}")
    print(f"Mean AUC de-meaned : {df.auc_demeaned.mean():.4f}")
    print(f"Mean drop from removing baseline-level info: {df.drop.mean():+.4f}")
    print("\nInterpretation: de-meaned AUC is the model's real within-patient discriminative")
    print("power -- the only kind that can't be a between-patient shortcut. The drop size is")
    print("how much of the intact-feature AUC was riding on 'which patient this typically is.'")


if __name__ == "__main__":
    main()
