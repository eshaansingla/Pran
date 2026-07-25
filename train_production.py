"""
train_production.py
===================
Train the FINAL production model (the honestly-best BALANCED variant) on ALL
available data and save everything predict.py / the frontend need.

Winning recipe (real data only — no SMOTE, no scale_pos_weight):
  negatives : hardware-normal windows
  positives : hardware-abnormal windows + exactly enough CHARIS-abnormal windows
              (randomly sampled) to make the two classes equal  -> 50:50
  normalise : domain-separated QuantileTransformer (qt_hw on hardware, qt_c on CHARIS)
  model     : XGBoost, scale_pos_weight = 1.0

Saved to models/production/:
  hybrid_balanced_xgb.json   trained booster
  qt_hw.pkl                  hardware-domain normaliser (fit on all 146 subjects)
  qt_c.pkl                   clinical-domain normaliser (fit on the sampled CHARIS set)
  metadata.json             features, thresholds, composition, validated LOPO metrics
"""
from __future__ import annotations
import json, pickle
from datetime import date
from pathlib import Path
import numpy as np
import xgboost as xgb

import hybrid_pipeline_v4 as H

OUT = Path("models/production"); OUT.mkdir(parents=True, exist_ok=True)
CHARIS_THR = 0.2953


def main():
    print("Loading data ...")
    c_y = np.load(H.CACHE_Y); c_X = np.load(H.CACHE_X)
    c_X_abn = c_X[c_y == 1]
    flags = json.load(open(H.FLAGS_PATH)); flags = flags.get("patients", flags)
    hw_X, hw_y, hw_pid, hw_sess, names = H.load_hw_labeled(H.HW_DIR, flags)
    n_hw_abn = int((hw_y == 1).sum()); n_hw_norm = int((hw_y == 0).sum())

    # Sample CHARIS abnormal to hit exact 50:50 (real data only)
    n_need = n_hw_norm - n_hw_abn
    rng = np.random.default_rng(H.SEED)
    sel = rng.choice(len(c_X_abn), size=n_need, replace=False)
    c_sample = c_X_abn[sel]
    print(f"Composition (50:50, no SMOTE):")
    print(f"  y=0 hardware normal   : {n_hw_norm:,}")
    print(f"  y=1 hardware abnormal : {n_hw_abn:,}")
    print(f"  y=1 CHARIS sampled    : {len(c_sample):,}")
    print(f"  positives={n_hw_abn+len(c_sample):,}  negatives={n_hw_norm:,}")

    device = H.get_device(); print(f"Device: {device.upper()}")

    qt_hw = H.fit_qt(hw_X); qt_c = H.fit_qt(c_sample)
    X_hw_q = qt_hw.transform(hw_X).astype(np.float32)
    X_c_q  = qt_c.transform(c_sample).astype(np.float32)
    X_tr = np.concatenate([X_hw_q, X_c_q])
    y_tr = np.concatenate([hw_y, np.ones(len(c_sample), dtype=hw_y.dtype)])

    idx = rng.permutation(len(y_tr)); cut = int(0.9 * len(idx))
    d_itr = xgb.DMatrix(X_tr[idx[:cut]], label=y_tr[idx[:cut]], feature_names=H.FEATURES)
    d_iva = xgb.DMatrix(X_tr[idx[cut:]], label=y_tr[idx[cut:]], feature_names=H.FEATURES)
    bst = xgb.train(H.xgb_params(device, 1.0, H.SEED), d_itr, num_boost_round=800,
                    evals=[(d_iva, "val")], early_stopping_rounds=50, verbose_eval=False)
    print(f"Trained. best_iteration={bst.best_iteration}")

    bst.save_model(str(OUT / "hybrid_balanced_xgb.json"))
    pickle.dump(qt_hw, open(OUT / "qt_hw.pkl", "wb"))
    pickle.dump(qt_c,  open(OUT / "qt_c.pkl",  "wb"))

    # Honest validated metrics come from the leakage-free LOPO run, not this fit
    m = json.load(open("results/hybrid_balanced/balanced_metrics.json"))["metrics"]
    gain = bst.get_score(importance_type="gain"); tot = sum(gain.values()) + 1e-12
    meta = {
        "model": "hybrid balanced (production)",
        "design": "real-data 50:50 (CHARIS abnormal sampled to balance), "
                  "domain-separated QT, no SMOTE, no scale_pos_weight",
        "date": date.today().isoformat(),
        "features": H.FEATURES, "fs": H.FS, "win": H.WIN, "step": H.STEP,
        "thresholds": {
            "patient_screening": CHARIS_THR,
            "window_balanced_best_f1": m["best_f1"]["threshold"],
            "window_high_recall_youden": m["at_youden_thr"]["threshold"],
        },
        "training_composition": {
            "hardware_normal": n_hw_norm, "hardware_abnormal": n_hw_abn,
            "charis_abnormal_sampled": int(len(c_sample)),
            "class_ratio": "1.00:1 (exact)",
        },
        "feature_importance": {f: round(gain.get(f, 0.0) / tot, 4) for f in H.FEATURES},
        "validated_lopo_metrics_146_subjects": {
            "auc": m["auc"], "avg_precision": m["avg_precision"],
            "f1": m["at_charis_thr"]["f1"], "precision": m["at_charis_thr"]["precision"],
            "recall": m["at_charis_thr"]["recall"],
            "balanced_accuracy": m["at_charis_thr"]["balanced_acc"],
            "mcc": m["at_charis_thr"]["mcc"], "best_f1": m["best_f1"]["f1"],
            "high_recall_option": {"recall": m["at_youden_thr"]["recall"],
                                   "threshold": m["at_youden_thr"]["threshold"]},
            "dose_response_rho_3level": m["dose_response_3level_rho"],
            "dose_response_monotonic_3level": m["dose_response_3level_mono"],
            "valsalva_pct": m["valsalva_pct"],
        },
        "clinical_foundation_model_lopo_auc": 0.9611,
        "honest_note": ("AUC (0.90) and the label-free within-subject dose-response "
                        "(rho 0.85, Valsalva 100%) are strong. F1 (~0.44) is modest "
                        "because the positive class is ~5% prevalent and hardware labels "
                        "are patient-level. This is a screening / ICP-trending proxy, "
                        "NOT a calibrated mmHg monitor or a diagnostic device."),
    }
    json.dump(meta, open(OUT / "metadata.json", "w"), indent=2)
    print(f"\nSaved production model -> {OUT}/")
    for k, v in meta["validated_lopo_metrics_146_subjects"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
