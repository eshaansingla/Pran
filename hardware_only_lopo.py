"""
hardware_only_lopo.py
=====================
NO CHARIS. Trains on the hardware data ONLY, using the real experimental
session labels (which manoeuvre the subject performed) as ground truth for the
ICP direction. Leakage-free LOPO over the 146 subjects.

Manoeuvre -> known ICP effect:
  head-up 30 (1) = lowest | supine (0) | head-down 10 (2) | Valsalva (3) = highest

Schemes tested:
  A  Valsalva (3)  vs  supine (0)         strongest contrast
  B  head-down (2) vs  head-up (1)        PURE POSTURAL — cleanest (no breathing/motion confound)
  C  high {2,3}    vs  low {0,1}          + within-subject dose-response of the model output

Output: results/hardware_only/hw_only_results.json
"""
from __future__ import annotations
import json, numpy as np
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import xgboost as xgb
import hybrid_pipeline_v4 as H

OUT = Path("results/hardware_only"); OUT.mkdir(parents=True, exist_ok=True)


def lopo_binary(X, sess, pid, pos_sessions, neg_sessions, device, label):
    """LOPO: train on other subjects' pos/neg windows, test on held-out subject's."""
    logo = LeaveOneGroupOut()
    keep = np.isin(sess, pos_sessions + neg_sessions)
    Xk, sk, pk = X[keep], sess[keep], pid[keep]
    yk = np.isin(sk, pos_sessions).astype(int)
    all_y, all_p = [], []
    for tr, te in logo.split(Xk, yk, pk):
        if len(np.unique(yk[tr])) < 2 or len(np.unique(yk[te])) < 1:
            continue
        qt = H.fit_qt(Xk[tr])
        Xtr = qt.transform(Xk[tr]).astype(np.float32)
        Xte = qt.transform(Xk[te]).astype(np.float32)
        spw = (yk[tr] == 0).sum() / max((yk[tr] == 1).sum(), 1)
        bst = xgb.train(H.xgb_params(device, spw, H.SEED),
                        xgb.DMatrix(Xtr, label=yk[tr], feature_names=H.FEATURES),
                        num_boost_round=200, verbose_eval=False)
        p = bst.predict(xgb.DMatrix(Xte, feature_names=H.FEATURES))
        all_y.extend(yk[te].tolist()); all_p.extend(p.tolist())
    auc = roc_auc_score(all_y, all_p) if len(set(all_y)) == 2 else float("nan")
    print(f"  [{label}] pooled LOPO AUC = {auc:.4f}  (n={len(all_y)} windows)")
    return round(float(auc), 4)


def lopo_doseresponse(X, sess, pid, device):
    """Train high{2,3} vs low{0,1} per fold; score ALL 4 sessions of held-out
    subject -> within-subject dose-response of a HARDWARE-ONLY model."""
    logo = LeaveOneGroupOut()
    order = [1, 0, 2, 3]
    persub, monos = [], 0
    for tr, te in logo.split(X, sess, pid):
        ytr = np.isin(sess[tr], [2, 3]).astype(int)
        if len(np.unique(ytr)) < 2:
            continue
        qt = H.fit_qt(X[tr])
        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        bst = xgb.train(H.xgb_params(device, spw, H.SEED),
                        xgb.DMatrix(qt.transform(X[tr]).astype(np.float32), label=ytr,
                                    feature_names=H.FEATURES),
                        num_boost_round=200, verbose_eval=False)
        Xte, ste = X[te], sess[te]
        p = bst.predict(xgb.DMatrix(qt.transform(Xte).astype(np.float32), feature_names=H.FEATURES))
        vec = [p[ste == k].mean() if (ste == k).sum() > 0 else np.nan for k in order]
        if all(np.isfinite(vec)):
            persub.append(vec)
            if vec[0] < vec[1] < vec[2] < vec[3]:
                monos += 1
    rhos = [spearmanr([0, 1, 2, 3], v)[0] for v in persub]
    rhos = [r for r in rhos if np.isfinite(r)]
    print(f"  [dose-response, hardware-only model] mean within-subject rho = {np.mean(rhos):+.3f}"
          f"  ({monos}/{len(persub)} strictly monotonic)")
    return {"mean_within_subject_rho": round(float(np.mean(rhos)), 3),
            "n_subjects": len(persub), "monotonic": monos}


def main():
    flags = json.load(open(H.FLAGS_PATH)); flags = flags.get("patients", flags)
    X, y, pid, sess, names = H.load_hw_labeled(H.HW_DIR, flags)
    device = H.get_device()
    print(f"Hardware-only (NO CHARIS). {len(np.unique(pid))} subjects, {len(X):,} windows. Device {device.upper()}\n")

    res = {}
    res["A_valsalva_vs_supine"]   = lopo_binary(X, sess, pid, [3], [0], device, "Valsalva vs supine")
    res["B_headdown_vs_headup"]   = lopo_binary(X, sess, pid, [2], [1], device, "head-down vs head-up (pure postural)")
    res["C_high_vs_low"]          = lopo_binary(X, sess, pid, [2, 3], [0, 1], device, "high{2,3} vs low{0,1}")
    res["C_dose_response"]        = lopo_doseresponse(X, sess, pid, device)

    json.dump(res, open(OUT / "hw_only_results.json", "w"), indent=2)
    print(f"\nSaved -> {OUT/'hw_only_results.json'}")


if __name__ == "__main__":
    main()
