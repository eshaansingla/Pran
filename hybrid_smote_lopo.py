"""
hybrid_smote_lopo.py
====================
Re-run the hybrid LOPO with SMOTE applied to the true minority — the
HARDWARE-abnormal windows (2,213 / 4.8%) — and compare head-to-head against the
no-SMOTE V4 model. SMOTE is applied INSIDE each fold, on training data only,
AFTER domain-separated QT normalisation, and ONLY within the hardware domain
(so it never interpolates between the clinical and optical sensors — which would
create unphysical cross-domain samples). The held-out patient is never touched.

Outputs:
  results/hybrid_smote/lopo_records_smote.pkl
  results/hybrid_smote/smote_comparison.json
"""
from __future__ import annotations
import json, pickle
from pathlib import Path
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, balanced_accuracy_score, accuracy_score,
    matthews_corrcoef, precision_recall_curve)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

import hybrid_pipeline_v4 as H   # reuse loaders, QT, params, dose-response

OUT = Path("results/hybrid_smote"); OUT.mkdir(parents=True, exist_ok=True)
CHARIS_THR = 0.2953
SEP = "=" * 66


def load_everything():
    c_X = np.load(H.CACHE_X); c_y = np.load(H.CACHE_Y); c_pid = np.load(H.CACHE_PID)
    flags = json.load(open(H.FLAGS_PATH))
    flags = flags.get("patients", flags)
    hw_X, hw_y, hw_pid, hw_sess, names = H.load_hw_labeled(H.HW_DIR, flags)
    n_hw_abn = int((hw_y == 1).sum()); n_hw_norm = int((hw_y == 0).sum())
    c_X_sel, c_y_sel, _, sel, n_c_abn = H.select_charis_patients(
        c_X, c_y, c_pid, n_hw_norm, n_hw_abn)
    return hw_X, hw_y, hw_pid, hw_sess, names, c_X_sel, c_y_sel


def run_lopo_smote(hw_X, hw_y, hw_pid, hw_sess, names, c_X, c_y, device):
    logo = LeaveOneGroupOut(); records = []
    print(f"\n[LOPO + SMOTE] {len(np.unique(hw_pid))} folds "
          f"(SMOTE hardware-abnormal -> balanced, within-domain, train-only)")
    for fold, (tr, te) in enumerate(logo.split(hw_X, hw_y, hw_pid)):
        X_hw_tr, y_hw_tr = hw_X[tr], hw_y[tr]
        X_hw_te, y_hw_te = hw_X[te], hw_y[te]

        qt_hw = H.fit_qt(X_hw_tr); qt_c = H.fit_qt(c_X)
        X_hw_tr_q = qt_hw.transform(X_hw_tr).astype(np.float32)
        X_hw_te_q = qt_hw.transform(X_hw_te).astype(np.float32)
        X_c_q     = qt_c.transform(c_X).astype(np.float32)

        # --- SMOTE the hardware domain only (abnormal up to normal count) ---
        n_abn = int((y_hw_tr == 1).sum())
        if n_abn >= 6:  # enough minority to interpolate
            k = min(5, n_abn - 1)
            X_hw_res, y_hw_res = SMOTE(random_state=H.SEED, k_neighbors=k).fit_resample(
                X_hw_tr_q, y_hw_tr)
            X_hw_res = X_hw_res.astype(np.float32)
        else:
            X_hw_res, y_hw_res = X_hw_tr_q, y_hw_tr

        X_tr = np.concatenate([X_hw_res, X_c_q])
        y_tr = np.concatenate([y_hw_res, c_y])
        spw  = H.scale_pos_weight(y_tr)  # ~1.0 now that HW is balanced

        # inner split for early stopping
        rng = np.random.default_rng(H.SEED + fold)
        idx = rng.permutation(len(y_tr)); cut = int(0.9 * len(idx))
        d_itr = xgb.DMatrix(X_tr[idx[:cut]], label=y_tr[idx[:cut]], feature_names=H.FEATURES)
        d_iva = xgb.DMatrix(X_tr[idx[cut:]], label=y_tr[idx[cut:]], feature_names=H.FEATURES)
        bst = xgb.train(H.xgb_params(device, spw, H.SEED + fold), d_itr,
                        num_boost_round=600, evals=[(d_iva, "v")],
                        early_stopping_rounds=50, verbose_eval=False)

        probs = bst.predict(xgb.DMatrix(X_hw_te_q, feature_names=H.FEATURES))
        test_pid = int(hw_pid[te[0]])
        records.append({"pid": test_pid,
                        "name": names[test_pid] if test_pid < len(names) else f"pid{test_pid}",
                        "true_label": int(y_hw_te[0]),
                        "y": y_hw_te.tolist(), "probs": probs.tolist(),
                        "mean_prob": round(float(probs.mean()), 4),
                        "sessions": hw_sess[te].tolist()})
        if fold % 25 == 0:
            print(f"  fold {fold:3d}/{len(np.unique(hw_pid))}  n_abn_train={n_abn}"
                  f" -> resampled {int((y_hw_res==1).sum())}")
    return records


def full_metrics(recs, tag):
    y = np.array([v for r in recs for v in r["y"]])
    p = np.array([v for r in recs for v in r["probs"]])
    auc = roc_auc_score(y, p); ap = average_precision_score(y, p)
    yhat = (p >= CHARIS_THR).astype(int)
    prec, rec, thr = precision_recall_curve(y, p)
    f1s = 2*prec*rec/(prec+rec+1e-12); bi = int(np.nanargmax(f1s))
    m = {
        "auc": round(float(auc), 4), "avg_precision": round(float(ap), 4),
        "at_charis_thr": {
            "accuracy": round(accuracy_score(y, yhat), 4),
            "balanced_acc": round(balanced_accuracy_score(y, yhat), 4),
            "f1": round(f1_score(y, yhat, zero_division=0), 4),
            "precision": round(precision_score(y, yhat, zero_division=0), 4),
            "recall": round(recall_score(y, yhat, zero_division=0), 4),
            "specificity": round(recall_score(1-y, 1-yhat, zero_division=0), 4),
            "mcc": round(matthews_corrcoef(y, yhat), 4)},
        "best_f1": {"threshold": round(float(thr[bi]), 4), "f1": round(float(f1s[bi]), 4),
                    "precision": round(float(prec[bi]), 4), "recall": round(float(rec[bi]), 4)},
    }
    dose = H.dose_response_analysis(recs)
    m["dose_response_3level_rho"] = dose.get("three_level", {}).get("mean_within_subject_spearman")
    m["dose_response_3level_mono"] = dose.get("three_level", {}).get("monotonic_fraction")
    m["dose_response_4level_rho"] = dose.get("mean_within_subject_spearman")
    val = H.valsalva_analysis(recs)
    m["valsalva_pct"] = val.get("pct_higher")
    print(f"\n  [{tag}] AUC={m['auc']} AP={m['avg_precision']} | "
          f"@thr F1={m['at_charis_thr']['f1']} P={m['at_charis_thr']['precision']} "
          f"R={m['at_charis_thr']['recall']} | bestF1={m['best_f1']['f1']} | "
          f"doseρ(3)={m['dose_response_3level_rho']} valsalva={m['valsalva_pct']}%")
    return m


def main():
    device = H.get_device()
    print(SEP); print("  HYBRID + SMOTE (hardware-minority) vs NO-SMOTE V4"); print(SEP)
    hw_X, hw_y, hw_pid, hw_sess, names, c_X, c_y = load_everything()

    recs_smote = run_lopo_smote(hw_X, hw_y, hw_pid, hw_sess, names, c_X, c_y, device)
    pickle.dump(recs_smote, open(OUT / "lopo_records_smote.pkl", "wb"))

    print("\n" + SEP + "\n  METRIC COMPARISON\n" + SEP)
    m_smote = full_metrics(recs_smote, "SMOTE")
    base = pickle.load(open("results/hybrid_pipeline_v4/lopo_records.pkl", "rb"))
    m_base = full_metrics(base, "NO-SMOTE (V4)")

    out = {"no_smote_v4": m_base, "smote_hardware_minority": m_smote}
    json.dump(out, open(OUT / "smote_comparison.json", "w"), indent=2)
    print(f"\n  Saved -> {OUT/'smote_comparison.json'}")


if __name__ == "__main__":
    main()
