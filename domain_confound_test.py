"""
domain_confound_test.py
=======================
Tests the proposed design: positives = CHARIS abnormal, negatives = hardware
normal (balanced, domain-separated QT). Question: does the model learn ICP, or
does it just learn "which sensor produced this" (domain confound)?

Decisive check: on HELD-OUT CHARIS patients, can it separate CHARIS-abnormal
from CHARIS-NORMAL? Both are the clinical sensor, so a domain detector cannot;
only a real ICP detector can.
"""
from __future__ import annotations
import json, numpy as np
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import hybrid_pipeline_v4 as H

SEED = 42


def main():
    c_X = np.load(H.CACHE_X); c_y = np.load(H.CACHE_Y); c_pid = np.load(H.CACHE_PID)
    flags = json.load(open(H.FLAGS_PATH)); flags = flags.get("patients", flags)
    hw_X, hw_y, hw_pid, hw_sess, names = H.load_hw_labeled(H.HW_DIR, flags)
    hw_norm = hw_X[hw_y == 0]

    # Hold out 4 CHARIS patients (that have BOTH abnormal and normal windows)
    rng = np.random.default_rng(SEED)
    good = [p for p in np.unique(c_pid)
            if ((c_pid == p) & (c_y == 1)).sum() > 200 and ((c_pid == p) & (c_y == 0)).sum() > 200]
    test_pids = list(rng.choice(good, size=min(4, len(good)), replace=False))
    te_mask = np.isin(c_pid, test_pids)
    tr_mask = ~te_mask
    print(f"Held-out CHARIS test patients: {test_pids}")

    c_abn_tr = c_X[tr_mask & (c_y == 1)]
    # balance: sample CHARIS-abn positives to match hardware-normal negatives
    n = min(len(c_abn_tr), len(hw_norm))
    c_abn_tr = c_abn_tr[rng.choice(len(c_abn_tr), n, replace=False)]
    hw_norm_s = hw_norm[rng.choice(len(hw_norm), n, replace=False)]

    # domain-separated QT
    qt_c = H.fit_qt(c_abn_tr); qt_h = H.fit_qt(hw_norm_s)
    X_tr = np.concatenate([qt_c.transform(c_abn_tr), qt_h.transform(hw_norm_s)]).astype(np.float32)
    y_tr = np.concatenate([np.ones(n), np.zeros(n)])
    print(f"Train: {n} CHARIS-abnormal (y=1)  vs  {n} hardware-normal (y=0)  [balanced]")

    device = H.get_device()
    idx = rng.permutation(len(y_tr)); cut = int(0.9 * len(idx))
    d_itr = xgb.DMatrix(X_tr[idx[:cut]], label=y_tr[idx[:cut]], feature_names=H.FEATURES)
    d_iva = xgb.DMatrix(X_tr[idx[cut:]], label=y_tr[idx[cut:]], feature_names=H.FEATURES)
    bst = xgb.train(H.xgb_params(device, 1.0, SEED), d_itr, num_boost_round=400,
                    evals=[(d_iva, "v")], early_stopping_rounds=40, verbose_eval=False)

    # --- DECISIVE TEST: held-out CHARIS abnormal vs CHARIS normal ---
    Xc_abn = c_X[te_mask & (c_y == 1)]
    Xc_nrm = c_X[te_mask & (c_y == 0)]
    # normalise with the CLINICAL-domain QT (these are clinical-sensor data)
    p_abn = bst.predict(xgb.DMatrix(qt_c.transform(Xc_abn), feature_names=H.FEATURES))
    p_nrm = bst.predict(xgb.DMatrix(qt_c.transform(Xc_nrm), feature_names=H.FEATURES))
    y = np.concatenate([np.ones(len(p_abn)), np.zeros(len(p_nrm))])
    p = np.concatenate([p_abn, p_nrm])
    auc_icp = roc_auc_score(y, p)

    print("\n" + "=" * 60)
    print("  DECISIVE TEST — held-out CHARIS: abnormal vs NORMAL")
    print("=" * 60)
    print(f"  mean score CHARIS-abnormal : {p_abn.mean():.3f}")
    print(f"  mean score CHARIS-normal   : {p_nrm.mean():.3f}")
    print(f"  AUC (can it tell them apart): {auc_icp:.3f}")
    print()
    if auc_icp > 0.75:
        print("  --> Learned real ICP contrast (separates abn vs norm same-sensor). OK-ish.")
    elif p_nrm.mean() > 0.5:
        print("  --> DOMAIN CONFOUND: calls CHARIS-NORMAL 'abnormal' too.")
        print("      The model learned 'clinical sensor = positive', not ICP.")
    else:
        print("  --> Weak/ambiguous ICP separation; likely domain-driven.")

    # For contrast: pure-CHARIS model (trained abn vs norm within CHARIS)
    try:
        pc = xgb.Booster(); pc.load_model("models/xgb_qt.json")
        import pickle
        qtp = pickle.load(open("models/qt_scaler.pkl", "rb"))
        pa = pc.predict(xgb.DMatrix(qtp.transform(Xc_abn), feature_names=H.FEATURES))
        pn = pc.predict(xgb.DMatrix(qtp.transform(Xc_nrm), feature_names=H.FEATURES))
        auc_pure = roc_auc_score(np.concatenate([np.ones(len(pa)), np.zeros(len(pn))]),
                                 np.concatenate([pa, pn]))
        print(f"\n  For comparison — PURE-CHARIS model on same test: AUC={auc_pure:.3f}")
        print("  (trained abn-vs-norm WITHIN CHARIS -> no domain shortcut available)")
    except Exception as e:
        print(f"\n  (pure-CHARIS comparison skipped: {e})")


if __name__ == "__main__":
    main()
