"""
hybrid_balanced_lopo.py
=======================
Class balance achieved with REAL DATA ONLY — no SMOTE, no scale_pos_weight.

Design (user-specified):
  negatives (y=0) : hardware-normal windows
  positives (y=1) : hardware-abnormal windows  +  exactly enough CHARIS-abnormal
                    windows (randomly sampled) to make positives == negatives.
  => class imbalance = 0 (50:50), entirely from genuine recordings.

Normalisation : domain-separated QuantileTransformer (qt_hw / qt_c), fit per-fold
                on training data only.
Model         : XGBoost, scale_pos_weight = 1.0 (already balanced).
Evaluation    : leakage-free LOPO over the 146 hardware subjects; CHARIS re-sampled
                and re-balanced inside every fold; held-out subject never in train.

Outputs:
  results/hybrid_balanced/lopo_records_balanced.pkl
  results/hybrid_balanced/balanced_metrics.json
"""
from __future__ import annotations
import json, pickle
from pathlib import Path
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, balanced_accuracy_score, accuracy_score,
    matthews_corrcoef, precision_recall_curve, roc_curve)
import xgboost as xgb

import hybrid_pipeline_v4 as H

OUT = Path("results/hybrid_balanced"); OUT.mkdir(parents=True, exist_ok=True)
CHARIS_THR = 0.2953
SEP = "=" * 66


def sample_charis(c_X_abn, n_needed, seed):
    """Randomly draw exactly n_needed CHARIS abnormal windows (no replacement)."""
    if n_needed >= len(c_X_abn):
        return c_X_abn
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(c_X_abn), size=n_needed, replace=False)
    return c_X_abn[idx]


def run(hw_X, hw_y, hw_pid, hw_sess, names, c_X_abn, device):
    logo = LeaveOneGroupOut(); records = []
    n_pts = len(np.unique(hw_pid))
    print(f"\n[BALANCED LOPO] {n_pts} folds — real-data 50:50, no SMOTE, spw=1.0")
    for fold, (tr, te) in enumerate(logo.split(hw_X, hw_y, hw_pid)):
        X_hw_tr, y_hw_tr = hw_X[tr], hw_y[tr]
        X_hw_te, y_hw_te = hw_X[te], hw_y[te]

        n_neg  = int((y_hw_tr == 0).sum())
        n_hpos = int((y_hw_tr == 1).sum())
        n_need = max(n_neg - n_hpos, 0)                      # CHARIS to reach 50:50
        c_sample = sample_charis(c_X_abn, n_need, H.SEED + fold)

        qt_hw = H.fit_qt(X_hw_tr); qt_c = H.fit_qt(c_sample)
        X_hw_tr_q = qt_hw.transform(X_hw_tr).astype(np.float32)
        X_hw_te_q = qt_hw.transform(X_hw_te).astype(np.float32)
        X_c_q     = qt_c.transform(c_sample).astype(np.float32)

        X_tr = np.concatenate([X_hw_tr_q, X_c_q])
        y_tr = np.concatenate([y_hw_tr, np.ones(len(c_sample), dtype=y_hw_tr.dtype)])
        # positives = n_hpos + n_need == n_neg  -> exactly balanced

        rng = np.random.default_rng(H.SEED + fold)
        idx = rng.permutation(len(y_tr)); cut = int(0.9 * len(idx))
        d_itr = xgb.DMatrix(X_tr[idx[:cut]], label=y_tr[idx[:cut]], feature_names=H.FEATURES)
        d_iva = xgb.DMatrix(X_tr[idx[cut:]], label=y_tr[idx[cut:]], feature_names=H.FEATURES)
        bst = xgb.train(H.xgb_params(device, 1.0, H.SEED + fold), d_itr,
                        num_boost_round=600, evals=[(d_iva, "v")],
                        early_stopping_rounds=50, verbose_eval=False)

        probs = bst.predict(xgb.DMatrix(X_hw_te_q, feature_names=H.FEATURES))
        tpid = int(hw_pid[te[0]])
        records.append({"pid": tpid,
                        "name": names[tpid] if tpid < len(names) else f"pid{tpid}",
                        "true_label": int(y_hw_te[0]),
                        "y": y_hw_te.tolist(), "probs": probs.tolist(),
                        "mean_prob": round(float(probs.mean()), 4),
                        "sessions": hw_sess[te].tolist()})
        if fold % 25 == 0:
            print(f"  fold {fold:3d}/{n_pts}  neg={n_neg}  hw_pos={n_hpos}  "
                  f"charis_sampled={n_need}  (pos==neg={n_hpos+n_need==n_neg})")
    return records


def full_metrics(recs, tag):
    y = np.array([v for r in recs for v in r["y"]])
    p = np.array([v for r in recs for v in r["probs"]])
    auc = roc_auc_score(y, p); ap = average_precision_score(y, p)
    yhat = (p >= CHARIS_THR).astype(int)
    prec, rec, thr = precision_recall_curve(y, p)
    f1s = 2*prec*rec/(prec+rec+1e-12); bi = int(np.nanargmax(f1s))
    fpr, tpr, jt = roc_curve(y, p); j = jt[np.argmax(tpr - fpr)]
    yj = (p >= j).astype(int)
    m = {
        "auc": round(float(auc), 4), "avg_precision": round(float(ap), 4),
        "prevalence": round(float(y.mean()), 4),
        "at_charis_thr": {
            "threshold": CHARIS_THR,
            "accuracy": round(accuracy_score(y, yhat), 4),
            "balanced_acc": round(balanced_accuracy_score(y, yhat), 4),
            "f1": round(f1_score(y, yhat, zero_division=0), 4),
            "precision": round(precision_score(y, yhat, zero_division=0), 4),
            "recall": round(recall_score(y, yhat, zero_division=0), 4),
            "specificity": round(recall_score(1-y, 1-yhat, zero_division=0), 4),
            "mcc": round(matthews_corrcoef(y, yhat), 4)},
        "at_youden_thr": {
            "threshold": round(float(j), 4),
            "f1": round(f1_score(y, yj, zero_division=0), 4),
            "precision": round(precision_score(y, yj, zero_division=0), 4),
            "recall": round(recall_score(y, yj, zero_division=0), 4),
            "balanced_acc": round(balanced_accuracy_score(y, yj), 4)},
        "best_f1": {"threshold": round(float(thr[bi]), 4), "f1": round(float(f1s[bi]), 4),
                    "precision": round(float(prec[bi]), 4), "recall": round(float(rec[bi]), 4)},
    }
    dose = H.dose_response_analysis(recs); val = H.valsalva_analysis(recs)
    m["dose_response_3level_rho"]  = dose.get("three_level", {}).get("mean_within_subject_spearman")
    m["dose_response_3level_mono"] = dose.get("three_level", {}).get("monotonic_fraction")
    m["dose_response_4level_rho"]  = dose.get("mean_within_subject_spearman")
    m["valsalva_pct"] = val.get("pct_higher")
    print(f"\n  [{tag}] AUC={m['auc']} AP={m['avg_precision']} | @charis F1={m['at_charis_thr']['f1']} "
          f"P={m['at_charis_thr']['precision']} R={m['at_charis_thr']['recall']} "
          f"bAcc={m['at_charis_thr']['balanced_acc']} MCC={m['at_charis_thr']['mcc']} | "
          f"bestF1={m['best_f1']['f1']} | doseρ3={m['dose_response_3level_rho']} val={m['valsalva_pct']}%")
    return m


def main():
    device = H.get_device()
    print(SEP); print("  HYBRID — REAL-DATA 50:50 (no SMOTE, no scale_pos_weight)"); print(SEP)
    c_y = np.load(H.CACHE_Y); c_X = np.load(H.CACHE_X)
    c_X_abn = c_X[c_y == 1]
    flags = json.load(open(H.FLAGS_PATH)); flags = flags.get("patients", flags)
    hw_X, hw_y, hw_pid, hw_sess, names = H.load_hw_labeled(H.HW_DIR, flags)
    print(f"  hardware: {len(np.unique(hw_pid))} subjects  "
          f"abn={int((hw_y==1).sum())}  norm={int((hw_y==0).sum())}")
    print(f"  CHARIS abnormal pool: {len(c_X_abn):,} windows (sample per fold to balance)")
    print(f"  Device: {device.upper()}")

    recs = run(hw_X, hw_y, hw_pid, hw_sess, names, c_X_abn, device)
    pickle.dump(recs, open(OUT / "lopo_records_balanced.pkl", "wb"))

    print("\n" + SEP + "\n  RESULTS — balanced (real 50:50)\n" + SEP)
    m = full_metrics(recs, "BALANCED")
    json.dump({"design": "real-data 50:50, CHARIS abnormal sampled to balance, "
               "no SMOTE, no scale_pos_weight, domain-separated QT",
               "metrics": m}, open(OUT / "balanced_metrics.json", "w"), indent=2)
    print(f"\n  Saved -> {OUT/'balanced_metrics.json'}")


if __name__ == "__main__":
    main()
