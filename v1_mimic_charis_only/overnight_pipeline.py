"""
overnight_pipeline.py
=====================
AUTONOMOUS OVERNIGHT PIPELINE — Extract max patients, retrain everything, validate.

Runs END-TO-END without human intervention:
  1. Full MIMIC-III scan (--scan_step 1) to extract ALL available ICP patients
  2. Clean extracted data (NaN, MAP clamping)
  3. Retrain XGBoost with 5-fold CV + bootstrapped CIs
  4. Run baseline comparison (LR, RF, SVM, threshold)
  5. Recompute LSTM metrics (per-sequence + per-horizon)
  6. Export firmware C header
  7. Run full test suite
  8. Generate final summary with all metrics

Usage:
    python overnight_pipeline.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR   = Path("results/binary")
MODELS_DIR    = Path("models")
LOG_FILE      = Path("overnight_pipeline.log")


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_step(name: str, cmd: list[str], timeout: int = 14400) -> bool:
    log(f"{'='*60}")
    log(f"STEP: {name}")
    log(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd, cwd=str(Path.cwd()), timeout=timeout,
            capture_output=True, text=True, env=os.environ.copy(),
        )
        output = (result.stdout or "") + (result.stderr or "")
        log(f"  Exit code: {result.returncode}")
        for line in output[-3000:].split("\n")[-40:]:
            if line.strip():
                log(f"  | {line.strip()}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        log(f"  TIMEOUT after {timeout}s")
        return False
    except Exception as e:
        log(f"  ERROR: {e}")
        return False


def step1_extract_more_patients():
    """Full scan of MIMIC-III for ALL patients with ICP signals."""
    log("PHASE 1: Extracting maximum MIMIC-III patients (FULL SCAN)")

    try:
        mi_pid = np.load(PROCESSED_DIR / "mimic_patient_ids.npy")
        log(f"  Current MIMIC patients: {len(set(mi_pid))}")
    except Exception:
        log("  No existing MIMIC data found")

    # Backup current data
    backup_dir = PROCESSED_DIR / "backup_pre_overnight"
    backup_dir.mkdir(exist_ok=True)
    for f in PROCESSED_DIR.glob("mimic_*.npy"):
        shutil.copy(f, backup_dir / f.name)
    log("  Backed up current MIMIC data")

    # Full scan — target 500 patients, scan_step=1 (every record)
    success = run_step(
        "MIMIC-III Full Extraction (scan_step=1, target=500)",
        [sys.executable, "build_mimic_features.py",
         "--target_patients", "500",
         "--scan_step", "1"],
        timeout=14400,  # 4 hours max
    )

    if success:
        try:
            new_pid = np.load(PROCESSED_DIR / "mimic_patient_ids.npy")
            new_mi = np.load(PROCESSED_DIR / "mimic_features.npy")
            log(f"  After extraction: {new_mi.shape[0]:,} windows, "
                f"{len(set(new_pid))} patients")
        except Exception:
            log("  Could not load new data")
    else:
        log("  Extraction failed — restoring backup")
        for f in backup_dir.glob("mimic_*.npy"):
            shutil.copy(f, PROCESSED_DIR / f.name)
    return success


def step2_clean_data():
    """Clean all data: NaN imputation, MAP clamping."""
    log("PHASE 2: Cleaning data")
    try:
        mi = np.load(PROCESSED_DIR / "mimic_features.npy").astype(np.float32)
        n_cols = mi.shape[1]
        map_col = 5 if n_cols >= 6 else n_cols - 1

        nan_before = int(np.isnan(mi).sum())
        log(f"  Shape: {mi.shape}, NaN: {nan_before}")

        # Impute NaN
        for j in range(mi.shape[1]):
            mask = np.isnan(mi[:, j])
            if mask.any():
                mi[mask, j] = float(np.nanmedian(mi[:, j]))
                log(f"  Imputed {mask.sum()} NaN in column {j}")

        # Clamp MAP
        bad_map = ((mi[:, map_col] < 40) | (mi[:, map_col] > 200)).sum()
        mi[:, map_col] = np.clip(mi[:, map_col], 40.0, 200.0)
        log(f"  Clamped {bad_map} MAP values to [40, 200]")

        np.save(PROCESSED_DIR / "mimic_features.npy", mi)
        log(f"  Saved. NaN remaining: {int(np.isnan(mi).sum())}")
        return True
    except Exception as e:
        log(f"  ERROR: {e}")
        return False


def step3_retrain_xgboost():
    """Retrain XGBoost with 5-fold patient-level CV."""
    log("PHASE 3: XGBoost retraining with 5-fold CV")
    success = run_step(
        "XGBoost Training",
        [sys.executable, "train_binary.py",
         "--processed_dir", str(PROCESSED_DIR),
         "--out_dir", str(RESULTS_DIR)],
        timeout=1800,
    )
    if success:
        try:
            meta = json.loads((MODELS_DIR / "binary_meta.json").read_text())
            log(f"  Test F1:  {meta['metrics']['f1']:.4f}")
            log(f"  Test AUC: {meta['metrics']['auc']:.4f}")
            log(f"  CV F1:    {meta['cross_validation']['f1_mean']} "
                f"± {meta['cross_validation']['f1_std']}")
            log(f"  CV AUC:   {meta['cross_validation']['auc_mean']} "
                f"± {meta['cross_validation']['auc_std']}")
        except Exception:
            pass
    return success


def step4_baselines():
    log("PHASE 4: Baseline comparison")
    return run_step("Baselines", [sys.executable, "baseline_comparison.py"], timeout=600)


def step5_lstm_recompute():
    log("PHASE 5: LSTM per-sequence + per-horizon recomputation")
    if not (MODELS_DIR / "lstm_forecast_v1.h5").exists():
        log("  LSTM model not found — skipping")
        return False
    return run_step("LSTM Metrics", [sys.executable, "recompute_lstm_metrics.py"], timeout=600)


def step6_firmware():
    log("PHASE 6: Firmware export")
    return run_step("Firmware", [sys.executable, "export_to_c.py"], timeout=60)


def step7_tests():
    log("PHASE 7: Test suite")
    return run_step("Tests", [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"], timeout=120)


def step8_summary():
    log("PHASE 8: Final summary")
    summary = {"completed": datetime.now().isoformat(), "data": {}, "xgboost": {}, "lstm": {}}

    try:
        ch = np.load(PROCESSED_DIR / "features.npy")
        mi = np.load(PROCESSED_DIR / "mimic_features.npy")
        ch_pid = np.load(PROCESSED_DIR / "patient_ids.npy")
        mi_pid = np.load(PROCESSED_DIR / "mimic_patient_ids.npy")
        summary["data"] = {
            "charis_patients": int(len(set(ch_pid))),
            "charis_windows": int(ch.shape[0]),
            "mimic_patients": int(len(set(mi_pid))),
            "mimic_windows": int(mi.shape[0]),
            "total_patients": int(len(set(ch_pid)) + len(set(mi_pid))),
            "total_windows": int(ch.shape[0] + mi.shape[0]),
        }
    except Exception:
        pass

    try:
        meta = json.loads((MODELS_DIR / "binary_meta.json").read_text())
        summary["xgboost"] = {
            "test_f1": meta["metrics"]["f1"],
            "test_auc": meta["metrics"]["auc"],
            "cv_f1_mean": meta["cross_validation"]["f1_mean"],
            "cv_f1_std": meta["cross_validation"]["f1_std"],
            "cv_auc_mean": meta["cross_validation"]["auc_mean"],
            "cv_auc_std": meta["cross_validation"]["auc_std"],
        }
    except Exception:
        pass

    try:
        lstm = json.loads(Path("results/lstm/honest_metrics.json").read_text())
        summary["lstm"] = {
            "per_sequence_auc": lstm["per_sequence_metrics"]["auc"],
            "per_sequence_f1": lstm["per_sequence_metrics"]["f1"],
            "auc_t1": lstm["per_horizon_metrics"][0]["auc"],
            "auc_t15": lstm["per_horizon_metrics"][-1]["auc"],
        }
    except Exception:
        pass

    Path("results").mkdir(exist_ok=True)
    Path("results/overnight_summary.json").write_text(json.dumps(summary, indent=2))

    log("\n" + "=" * 70)
    log("  ★ OVERNIGHT PIPELINE — FINAL RESULTS ★")
    log("=" * 70)
    d = summary.get("data", {})
    x = summary.get("xgboost", {})
    l = summary.get("lstm", {})
    log(f"  Data:  {d.get('total_patients','?')} patients, "
        f"{d.get('total_windows','?'):,} windows")
    log(f"  XGB Test  F1={x.get('test_f1','?'):.4f}  AUC={x.get('test_auc','?'):.4f}")
    log(f"  XGB CV    F1={x.get('cv_f1_mean','?')} ± {x.get('cv_f1_std','?')}")
    log(f"  XGB CV    AUC={x.get('cv_auc_mean','?')} ± {x.get('cv_auc_std','?')}")
    log(f"  LSTM Seq  AUC={l.get('per_sequence_auc','?')}  F1={l.get('per_sequence_f1','?')}")
    log(f"  LSTM t+1  AUC={l.get('auc_t1','?')}")
    log(f"  LSTM t+15 AUC={l.get('auc_t15','?')}")
    log("=" * 70)
    return True


def main():
    # Clear log
    LOG_FILE.write_text("")

    log("#" * 70)
    log("  AUTONOMOUS OVERNIGHT PIPELINE")
    log(f"  Started: {datetime.now().isoformat()}")
    log(f"  Credentials: {os.environ.get('PHYSIONET_USERNAME', 'MISSING')[:5]}...")
    log("#" * 70)

    if not os.environ.get("PHYSIONET_USERNAME"):
        log("ERROR: PHYSIONET_USERNAME not set!")
        sys.exit(1)

    results = {}
    results["1_extract"]   = step1_extract_more_patients()
    results["2_clean"]     = step2_clean_data()
    results["3_xgboost"]   = step3_retrain_xgboost()
    results["4_baselines"] = step4_baselines()
    results["5_lstm"]      = step5_lstm_recompute()
    results["6_firmware"]  = step6_firmware()
    results["7_tests"]     = step7_tests()
    results["8_summary"]   = step8_summary()

    log("\n" + "=" * 70)
    log("  STEP STATUS")
    for step, ok in results.items():
        log(f"  {step:<20} {'✅ PASS' if ok else '❌ FAIL'}")
    log(f"  Completed: {datetime.now().isoformat()}")
    log(f"  Full log: {LOG_FILE.absolute()}")
    log("=" * 70)


if __name__ == "__main__":
    main()
