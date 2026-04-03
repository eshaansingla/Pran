"""
run_pipeline.py
===============
Full ICP monitoring pipeline in one command.

Steps:
  1. Verify / extract CHARIS features
  2. Verify / extract MIMIC-III ICP features  (streaming, no bulk download)
  3. Combine datasets  (MIMIC -> train only; CHARIS -> train/val/test)
  4. Train XGBoost  (patient-level GroupShuffleSplit 70/10/20)
  5. Evaluate  (classification report, confusion matrix, ROC curves, SHAP)
  6. Save model + plots

Usage:
    python run_pipeline.py                 # full run
    python run_pipeline.py --skip_mimic    # skip MIMIC download (use cached)
    python run_pipeline.py --force         # re-extract even if cache exists
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Credentials ────────────────────────────────────────────────────────────────
os.environ["PHYSIONET_USERNAME"] = "eshaansingla2005"
os.environ["PHYSIONET_PASSWORD"] = "+5Q5,,jdcy_ty8"

# ── Directories ────────────────────────────────────────────────────────────────
PROCESSED_DIR  = Path("data/processed")
CHARIS_RAW_DIR = Path("data/raw/charis")
MODEL_DIR      = Path("models/xgboost_combined")
RESULTS_DIR    = Path("results")

CHARIS_MAX_PID = 13   # patient IDs 1-13 = CHARIS; 101+ = MIMIC

# ── Pretty helpers ─────────────────────────────────────────────────────────────
W = 64

def _hdr(title: str) -> None:
    print(f"\n{'='*W}")
    print(f"  {title}")
    print(f"{'='*W}")

def _sec(title: str) -> None:
    print(f"\n  {'-'*56}")
    print(f"  {title}")
    print(f"  {'-'*56}")

def _ok(msg: str) -> None:  print(f"  [OK]  {msg}")
def _inf(msg: str) -> None: print(f"  [..] {msg}")
def _warn(msg: str) -> None: print(f"  [!!] {msg}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 – CHARIS features
# ══════════════════════════════════════════════════════════════════════════════

def step_charis(force: bool) -> None:
    _sec("Step 1 / 5  |  CHARIS features")

    feat_path = PROCESSED_DIR / "features.npy"
    lab_path  = PROCESSED_DIR / "labels.npy"
    pid_path  = PROCESSED_DIR / "patient_ids.npy"

    if not force and feat_path.exists() and lab_path.exists() and pid_path.exists():
        import numpy as np
        f = np.load(feat_path)
        l = np.load(lab_path)
        p = np.load(pid_path)
        charis_pids = set(int(x) for x in p if int(x) <= CHARIS_MAX_PID)
        if len(charis_pids) > 0 and f.shape[0] == l.shape[0]:
            _ok(f"Cached  {f.shape[0]:,} windows | {len(charis_pids)} patients")
            return

    _inf("Extracting CHARIS features from raw waveforms ...")
    if not CHARIS_RAW_DIR.exists():
        _warn("CHARIS raw data not found. Run download_charis.py first.")
        sys.exit(1)

    import numpy as np
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from data.extract_features import extract_all_features
    from data.generate_labels import assign_label
    from data.segment_windows import segment_record

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    hea_files = sorted(CHARIS_RAW_DIR.rglob("*.hea"))
    _inf(f"Found {len(hea_files)} CHARIS records")

    all_feats, all_labs, all_pids = [], [], []
    for pid, hea in enumerate(hea_files, start=1):
        rec_path = str(hea.with_suffix(""))
        wins = []
        try:
            from data.segment_windows import segment_record as _seg
            for win, _pid, _st in _seg(rec_path, pid, "ICP"):
                wins.append(win)
        except Exception as e:
            _warn(f"  Skipping {hea.stem}: {e}")
            continue
        if not wins:
            continue
        for w in wins:
            feat  = extract_all_features(w)[:8]
            label = assign_label(float(__import__('numpy').nanmedian(w)))
            all_feats.append(feat)
            all_labs.append(label)
            all_pids.append(pid)
        _inf(f"  Patient {pid}: {len(wins)} windows")

    np.save(feat_path, np.vstack(all_feats).astype("float32"))
    np.save(lab_path,  np.array(all_labs, dtype="int64"))
    np.save(pid_path,  np.array(all_pids, dtype="int32"))
    _ok(f"Saved {len(all_feats):,} CHARIS windows")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 – MIMIC features (streaming)
# ══════════════════════════════════════════════════════════════════════════════

def step_mimic(force: bool, target_patients: int, scan_step: int) -> None:
    _sec("Step 2 / 5  |  MIMIC-III ICP features")

    mi_feat = PROCESSED_DIR / "mimic_features.npy"
    mi_lab  = PROCESSED_DIR / "mimic_labels.npy"
    mi_pid  = PROCESSED_DIR / "mimic_patient_ids.npy"

    if not force and mi_feat.exists() and mi_lab.exists() and mi_pid.exists():
        import numpy as np
        f = np.load(mi_feat)
        l = np.load(mi_lab)
        p = np.load(mi_pid)
        pids = set(p.tolist())
        _ok(f"Cached  {f.shape[0]:,} windows | {len(pids)} patients")
        return

    import requests, time as _time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import numpy as np

    AUTH     = (os.environ["PHYSIONET_USERNAME"], os.environ["PHYSIONET_PASSWORD"])
    BASE_URL = "https://physionet.org/files/mimic3wdb/1.0"
    ICP_NAMES = {"ICP"}

    def _get(url):
        for attempt in range(3):
            try:
                r = requests.get(url, auth=AUTH, timeout=20)
                return r if r.status_code == 200 else None
            except Exception:
                if attempt == 2: return None
                _time.sleep(1.0 * (attempt + 1))

    def _sigs(text):
        out = []
        for line in text.strip().split("\n")[1:]:
            p = line.strip().split()
            if p: out.append(p[-1].upper())
        return out

    def _resample(sig, fs):
        if fs == 125: return sig
        n = int(len(sig) * 125 / fs)
        return __import__('numpy').interp(
            __import__('numpy').linspace(0, 1, n),
            __import__('numpy').linspace(0, 1, len(sig)), sig
        ).astype("float32")

    def _valid(win):
        import numpy as np
        missing = np.isnan(win).mean()
        if missing > 0.30: return False
        clean = win[~np.isnan(win)]
        if len(clean) < 625: return False
        if np.any(clean < 0) or np.any(clean > 50): return False
        if clean.std() < 0.01: return False
        return True

    def _label(med):
        return 0 if med < 15 else (1 if med < 20 else 2)

    def _bandpass(sig, low, high):
        from scipy.signal import butter, sosfiltfilt
        nyq = 125 / 2.0
        lo, hi = max(low/nyq, 1e-4), min(high/nyq, 0.9999)
        if lo >= hi: return sig.copy()
        sos = butter(4, [lo, hi], btype="bandpass", output="sos")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return sosfiltfilt(sos, sig)

    def _feat8(win, abp=None):
        import numpy as np
        if np.any(np.isnan(win)):
            win = win.copy(); win[np.isnan(win)] = float(np.nanmedian(win))
        cb = _bandpass(win, 1.0, 2.0)
        ca = (float(np.percentile(cb,99)-np.percentile(cb,1)))*10.0
        fb = _bandpass(win, 0.7, 2.5)
        fq = np.fft.rfftfreq(len(fb), d=1/125)
        pw = np.abs(np.fft.rfft(fb))**2
        mk = (fq>=0.7)&(fq<=2.5)
        cf = float(fq[mk][np.argmax(pw[mk])]) if mk.any() else 1.0
        rb = _bandpass(win, 0.1, 0.5)
        ra = (float(np.percentile(rb,99)-np.percentile(rb,1)))*10.0
        try:
            import pywt
            n100 = int(len(win)*100/125)
            x100 = np.interp(np.linspace(0,1,n100),np.linspace(0,1,len(win)),win)
            coeffs = pywt.wavedec(x100,"db4",level=5)
            en = [float(np.sum(c**2)) for c in coeffs]
            tot = sum(en)+1e-10
            sp = float(np.clip(en[0]/tot,0,1)); cp2 = float(np.clip(en[1]/tot,0,1))
        except ImportError:
            fq2 = np.fft.rfftfreq(len(win),d=1/125)
            pw2 = np.abs(np.fft.rfft(win))**2
            tot = pw2.sum()+1e-10
            sp = float(pw2[fq2<=1.56].sum()/tot)
            cp2 = float(pw2[(fq2>1.56)&(fq2<=3.12)].sum()/tot)
        if abp is not None and not np.all(np.isnan(abp)):
            mv = float(np.nanmean(abp))
        else:
            sm = float(np.nanmean(win)); mv = sm if sm>50 else 90.0
        return np.array([ca,cf,ra,sp,cp2,mv,0.0,0.0],dtype="float32")

    def _extract_record(rec_dir):
        import numpy as np, wfdb
        rec_id = rec_dir.split("/")[-1]
        pn_dir = f"mimic3wdb/1.0/{rec_dir}"
        WS = 1250
        feats, labs = [], []
        for seg_suffix in ["_layout"] + [f"_{i:04d}" for i in range(1, 25)]:
            seg = rec_id + seg_suffix
            try:
                sh = wfdb.rdheader(seg, pn_dir=pn_dir)
                upper = [s.upper() for s in sh.sig_name]
                if "ICP" not in upper: continue
                icp_idx = upper.index("ICP")
                abp_idx = next((i for i,s in enumerate(upper)
                                if s in {"ABP","ART","AP","ABPM","ARTM"}), None)
                chans = [icp_idx] if abp_idx is None else [icp_idx, abp_idx]
                rec = wfdb.rdrecord(seg, pn_dir=pn_dir, channels=chans)
                fs  = int(rec.fs)
                icp_raw = rec.p_signal[:,0].astype("float32")
                abp_raw = rec.p_signal[:,1].astype("float32") if rec.p_signal.shape[1]>1 else None
                if fs != 125:
                    icp_raw = _resample(icp_raw, fs)
                    if abp_raw is not None: abp_raw = _resample(abp_raw, fs)
                for i in range(len(icp_raw)//WS):
                    s = i*WS
                    wi = icp_raw[s:s+WS].copy()
                    wa = abp_raw[s:s+WS].copy() if abp_raw is not None else None
                    if not _valid(wi): continue
                    feats.append(_feat8(wi, wa))
                    labs.append(_label(float(np.nanmedian(wi))))
            except Exception:
                continue
            if feats: break
        return feats, labs

    # ---- Scan for ICP records ----
    _inf(f"Fetching MIMIC-III record list ...")
    resp = requests.get(f"{BASE_URL}/RECORDS", auth=AUTH, timeout=30)
    if resp.status_code != 200:
        _warn(f"Cannot access MIMIC-III (HTTP {resp.status_code})")
        _warn("Continuing with CHARIS only.")
        return

    all_recs = [l.rstrip("/") for l in resp.text.strip().split("\n")]
    candidates = all_recs[::scan_step]
    _inf(f"Scanning {len(candidates)} headers for ICP signal ...")

    def _check(rec_dir):
        rec_id = rec_dir.split("/")[-1]
        r = _get(f"{BASE_URL}/{rec_dir}/{rec_id}_layout.hea")
        if r is None:
            r = _get(f"{BASE_URL}/{rec_dir}/{rec_id}_0001.hea")
        if r is None: return rec_dir, False
        return rec_dir, bool(ICP_NAMES & set(_sigs(r.text)))

    icp_dirs = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_check, rec): rec for rec in candidates}
        done = 0
        for fut in as_completed(futs):
            rec_dir, has_icp = fut.result()
            done += 1
            if has_icp:
                icp_dirs.append(rec_dir)
                print(f"  [+] ICP: {rec_dir}  ({len(icp_dirs)}/{target_patients})",
                      flush=True)
            if done % 500 == 0:
                _inf(f"{done}/{len(candidates)} scanned  |  {len(icp_dirs)} ICP found")
            if len(icp_dirs) >= target_patients:
                pool.shutdown(wait=False, cancel_futures=True)
                break

    icp_dirs = icp_dirs[:target_patients]
    _inf(f"Found {len(icp_dirs)} ICP records. Extracting features ...")

    import numpy as np
    all_feats, all_labs, all_pids = [], [], []
    for i, rec_dir in enumerate(icp_dirs):
        pid = 100 + i + 1
        print(f"  [{i+1:02d}/{len(icp_dirs)}] {rec_dir} ... ", end="", flush=True)
        feats, labs = _extract_record(rec_dir)
        if feats:
            c = {0:labs.count(0),1:labs.count(1),2:labs.count(2)}
            print(f"{len(feats)} windows  N={c[0]} E={c[1]} C={c[2]}")
            all_feats.extend(feats)
            all_labs.extend(labs)
            all_pids.extend([pid]*len(feats))
        else:
            print("no valid windows")

    if not all_feats:
        _warn("No MIMIC windows extracted. Continuing with CHARIS only.")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(mi_feat, np.vstack(all_feats).astype("float32"))
    np.save(mi_lab,  np.array(all_labs, dtype="int64"))
    np.save(mi_pid,  np.array(all_pids, dtype="int32"))

    lc = __import__('numpy').bincount(np.array(all_labs), minlength=3)
    _ok(f"Saved {len(all_feats):,} MIMIC windows  |  "
        f"N={lc[0]:,}  E={lc[1]:,}  C={lc[2]:,}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3+4 – Combine & Train
# ══════════════════════════════════════════════════════════════════════════════

def step_train(out_dir: Path) -> dict:
    _sec("Steps 3-4 / 5  |  Combine datasets + Train XGBoost")

    import numpy as np
    from sklearn.model_selection import GroupShuffleSplit

    # ---- Load ----
    ch_feat = np.load(PROCESSED_DIR / "features.npy")[:, :8]
    ch_lab  = np.load(PROCESSED_DIR / "labels.npy")
    ch_pid  = np.load(PROCESSED_DIR / "patient_ids.npy").astype("int32")

    # Only CHARIS rows
    mask = ch_pid <= CHARIS_MAX_PID
    ch_feat, ch_lab, ch_pid = ch_feat[mask], ch_lab[mask], ch_pid[mask]
    _ok(f"CHARIS  {ch_feat.shape[0]:>9,} windows | {len(set(ch_pid.tolist()))} patients")

    mi_path = PROCESSED_DIR / "mimic_features.npy"
    if mi_path.exists():
        mi_feat = np.load(mi_path)[:, :8]
        mi_lab  = np.load(PROCESSED_DIR / "mimic_labels.npy")
        mi_pid  = np.load(PROCESSED_DIR / "mimic_patient_ids.npy").astype("int32")
        _ok(f"MIMIC   {mi_feat.shape[0]:>9,} windows | {len(set(mi_pid.tolist()))} patients"
            " (train only)")
        has_mimic = True
    else:
        has_mimic = False
        _warn("No MIMIC data found — training on CHARIS only")

    # ---- CHARIS split ----
    SEED = 42
    gss1 = GroupShuffleSplit(1, test_size=0.20, random_state=SEED)
    tv, te = next(gss1.split(ch_feat, ch_lab, groups=ch_pid))
    X_tv, y_tv, g_tv = ch_feat[tv], ch_lab[tv], ch_pid[tv]
    X_test, y_test    = ch_feat[te], ch_lab[te]

    gss2 = GroupShuffleSplit(1, test_size=0.125, random_state=SEED)
    tr, va = next(gss2.split(X_tv, y_tv, groups=g_tv))
    X_val, y_val = X_tv[va], y_tv[va]
    X_tr0, y_tr0 = X_tv[tr], y_tv[tr]

    # Augment train with MIMIC
    if has_mimic:
        X_train = np.vstack([X_tr0, mi_feat])
        y_train = np.concatenate([y_tr0, mi_lab])
    else:
        X_train, y_train = X_tr0, y_tr0

    total_pats = len(set(ch_pid.tolist())) + (len(set(mi_pid.tolist())) if has_mimic else 0)
    _inf(f"Train {len(X_train):,}  Val {len(X_val):,}  Test {len(X_test):,}  "
         f"(total patients: {total_pats})")

    # ---- Class weights ----
    cc = np.bincount(y_train.astype(int), minlength=3).astype(float)
    tot = cc.sum()
    w = np.select([y_train==0, y_train==1], [tot/(3*cc[0]), tot/(3*cc[1])],
                  default=tot/(3*cc[2]))

    # ---- Train ----
    import xgboost as xgb
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w)
    dval   = xgb.DMatrix(X_val,   label=y_val)
    dtest  = xgb.DMatrix(X_test,  label=y_test)

    params = dict(objective="multi:softmax", num_class=3,
                  eval_metric="mlogloss", eta=0.05,
                  max_depth=6, min_child_weight=5,
                  subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=1.0, alpha=0.1,
                  random_state=SEED, tree_method="hist", verbosity=0)

    print(f"\n  Training ... ", end="", flush=True)
    t0 = time.time()
    evals_result: dict = {}
    bst = xgb.train(params, dtrain, num_boost_round=2000,
                    evals=[(dtrain,"train"),(dval,"val")],
                    early_stopping_rounds=50, verbose_eval=False,
                    evals_result=evals_result)
    elapsed = time.time() - t0
    print(f"done  ({elapsed:.0f}s, best_iter={bst.best_iteration})")

    # ---- Evaluate ----
    from sklearn.metrics import (classification_report, balanced_accuracy_score,
                                  f1_score, confusion_matrix, roc_auc_score)
    from sklearn.preprocessing import label_binarize

    y_pred = bst.predict(dtest).astype(int)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    bal_acc  = balanced_accuracy_score(y_test, y_pred)
    report   = classification_report(
        y_test, y_pred,
        target_names=["Normal", "Elevated", "Critical"],
        output_dict=False,
    )
    report_dict = classification_report(
        y_test, y_pred,
        target_names=["Normal", "Elevated", "Critical"],
        output_dict=True,
    )
    cm = confusion_matrix(y_test, y_pred)

    # AUC (need proba) — use softprob
    params_prob = dict(params, objective="multi:softprob")
    bst_prob = xgb.train(params_prob, dtrain, num_boost_round=bst.best_iteration+1,
                         evals=[(dval,"val")], verbose_eval=False)
    y_prob = bst_prob.predict(dtest).reshape(-1, 3)
    y_bin  = label_binarize(y_test, classes=[0,1,2])
    try:
        auc = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
    except Exception:
        auc = float("nan")

    # Save model
    import pickle
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "xgboost_final.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(bst, fh)

    return dict(bst=bst, bst_prob=bst_prob, y_test=y_test, y_pred=y_pred,
                y_prob=y_prob, macro_f1=macro_f1, bal_acc=bal_acc,
                report=report, report_dict=report_dict,
                cm=cm, auc=auc, model_path=model_path,
                X_train=X_train, y_train=y_train,
                has_mimic=has_mimic,
                n_charis_pats=len(set(ch_pid.tolist())),
                n_mimic_pats=len(set(mi_pid.tolist())) if has_mimic else 0,
                n_charis_win=ch_feat.shape[0],
                n_mimic_win=mi_feat.shape[0] if has_mimic else 0)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 – Plots & report
# ══════════════════════════════════════════════════════════════════════════════

def step_report(results: dict, out_dir: Path) -> None:
    _sec("Step 5 / 5  |  Plots & summary")

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    from sklearn.preprocessing import label_binarize

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    labels_str = ["Normal", "Elevated", "Critical"]
    cm     = results["cm"]
    y_test = results["y_test"]
    y_prob = results["y_prob"]

    # ── Confusion matrix ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ticks = range(3)
    ax.set(xticks=list(ticks), yticks=list(ticks),
           xticklabels=labels_str, yticklabels=labels_str,
           xlabel="Predicted", ylabel="True",
           title="Confusion Matrix")
    thresh = cm.max() / 2.0
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black", fontsize=12)
    plt.tight_layout()
    cm_path = RESULTS_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    _ok(f"Confusion matrix -> {cm_path}")

    # ── ROC curves ────────────────────────────────────────────────────────────
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    colors = ["steelblue", "darkorange", "forestgreen"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (lbl, col) in enumerate(zip(labels_str, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        from sklearn.metrics import auc as _auc
        a = _auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{lbl}  (AUC = {a:.3f})")
    ax.plot([0,1],[0,1],"k--",lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    roc_path = RESULTS_DIR / "roc_curves.png"
    fig.savefig(roc_path, dpi=120)
    plt.close(fig)
    _ok(f"ROC curves       -> {roc_path}")

    # ── SHAP importance ───────────────────────────────────────────────────────
    try:
        import shap
        import numpy as np
        FEAT_NAMES = ["cardiac_amp","cardiac_freq","resp_amp","slow_wave_pwr",
                      "cardiac_pwr","MAP","head_angle","motion_flag"]
        bst = results["bst"]
        bg  = results["X_train"][
            np.random.default_rng(42).choice(len(results["X_train"]),
                                             min(500, len(results["X_train"])),
                                             replace=False)
        ]
        explainer = shap.TreeExplainer(bst)
        shap_exp  = explainer(bg)
        sv = shap_exp.values
        if sv.ndim == 3:
            mean_abs = np.abs(sv).mean(axis=(0, 2))
        else:
            mean_abs = np.abs(sv).mean(axis=0)

        order = np.argsort(mean_abs)[::-1]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh([FEAT_NAMES[i] for i in order][::-1],
                mean_abs[order][::-1], color="steelblue")
        ax.set(xlabel="Mean |SHAP value|", title="Feature Importance (SHAP)")
        plt.tight_layout()
        shap_path = RESULTS_DIR / "shap_importance.png"
        fig.savefig(shap_path, dpi=120)
        plt.close(fig)
        _ok(f"SHAP importance  -> {shap_path}")
    except Exception as e:
        _warn(f"SHAP skipped: {e}")

    # ── Text report ───────────────────────────────────────────────────────────
    rd     = results["report_dict"]
    f1_n   = rd["Normal"]["f1-score"]
    f1_e   = rd["Elevated"]["f1-score"]
    f1_c   = rd["Critical"]["f1-score"]
    status = "PASS" if results["macro_f1"] >= 0.70 else "FAIL"
    gap    = results["macro_f1"] - 0.70
    gap_s  = f"+{gap:.4f}" if gap >= 0 else f"{gap:.4f}"

    report_txt = out_dir / "final_report.txt"
    lines = [
        f"{'='*W}",
        f"  ICP MONITORING  –  FINAL RESULTS  [{status}]",
        f"{'='*W}",
        f"",
        f"  Dataset",
        f"    CHARIS  : {results['n_charis_pats']:>3} patients  |  "
        f"{results['n_charis_win']:>9,} windows",
    ]
    if results["has_mimic"]:
        lines += [
            f"    MIMIC   : {results['n_mimic_pats']:>3} patients  |  "
            f"{results['n_mimic_win']:>9,} windows  (train only)",
        ]
    lines += [
        f"    Total   : {results['n_charis_pats']+results['n_mimic_pats']:>3} patients  |  "
        f"{results['n_charis_win']+results['n_mimic_win']:>9,} windows",
        f"",
        f"  Performance  (test = CHARIS hold-out)",
        f"    Macro F1-score  :  {results['macro_f1']:.4f}  (target >= 0.70  ->  {status}, gap {gap_s})",
        f"    Balanced Acc    :  {results['bal_acc']:.4f}",
        f"    Macro AUC (OvR) :  {results['auc']:.4f}",
        f"",
        f"  Per-class F1",
        f"    Normal    :  {f1_n:.4f}",
        f"    Elevated  :  {f1_e:.4f}  (bottleneck, target >= 0.55)",
        f"    Critical  :  {f1_c:.4f}",
        f"",
        f"  Classification Report",
        f"",
        results["report"],
        f"",
        f"  Model saved  :  {results['model_path']}",
        f"  Plots        :  {RESULTS_DIR}/",
        f"{'='*W}",
    ]
    report_txt.write_text("\n".join(lines), encoding="utf-8")
    _ok(f"Report text      -> {report_txt}")

    return lines


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ICP monitoring full pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--skip_mimic",     action="store_true",
                        help="Skip MIMIC download (use cached data/processed/mimic_*.npy)")
    parser.add_argument("--force",          action="store_true",
                        help="Re-extract features even if cache exists")
    parser.add_argument("--target_patients",type=int, default=50,
                        help="Number of MIMIC ICP patients to collect")
    parser.add_argument("--scan_step",      type=int, default=20,
                        help="Check every Nth MIMIC header (smaller = more thorough)")
    parser.add_argument("--out_dir",        type=Path,
                        default=Path("models/xgboost_combined"))
    args = parser.parse_args()

    t_start = time.time()
    _hdr("ICP MONITORING PIPELINE  |  Capstone 2026")

    print(f"\n  Dirs")
    print(f"    Processed data : {PROCESSED_DIR}/")
    print(f"    Models         : {args.out_dir}/")
    print(f"    Plots          : {RESULTS_DIR}/")

    # 1. CHARIS
    step_charis(args.force)

    # 2. MIMIC
    if args.skip_mimic:
        _sec("Step 2 / 5  |  MIMIC-III ICP features")
        _inf("--skip_mimic set: using cached files (if any)")
    else:
        step_mimic(args.force, args.target_patients, args.scan_step)

    # 3+4. Combine + Train
    results = step_train(args.out_dir)

    # 5. Plots + report
    report_lines = step_report(results, args.out_dir)

    # ── Final console summary ──────────────────────────────────────────────────
    rd     = results["report_dict"]
    f1_n   = rd["Normal"]["f1-score"]
    f1_e   = rd["Elevated"]["f1-score"]
    f1_c   = rd["Critical"]["f1-score"]
    status = "PASS" if results["macro_f1"] >= 0.70 else "FAIL"
    gap    = results["macro_f1"] - 0.70
    gap_s  = f"+{gap:.4f}" if gap >= 0 else f"{gap:.4f}"
    elapsed = time.time() - t_start

    _hdr(f"FINAL RESULTS  [{status}]")
    print(f"\n  Dataset")
    print(f"    CHARIS   {results['n_charis_pats']:>3} patients    {results['n_charis_win']:>9,} windows")
    if results["has_mimic"]:
        print(f"    MIMIC    {results['n_mimic_pats']:>3} patients    {results['n_mimic_win']:>9,} windows  (train only)")
    n_total_p = results["n_charis_pats"] + results["n_mimic_pats"]
    n_total_w = results["n_charis_win"]  + results["n_mimic_win"]
    print(f"    Total    {n_total_p:>3} patients    {n_total_w:>9,} windows")

    print(f"\n  Performance  (test set = CHARIS hold-out)")
    print(f"    Macro F1-score  :  {results['macro_f1']:.4f}   target >= 0.70  ->  {status}  ({gap_s})")
    print(f"    Balanced Acc    :  {results['bal_acc']:.4f}")
    print(f"    Macro AUC (OvR) :  {results['auc']:.4f}")

    print(f"\n  Per-class F1")
    print(f"    Normal    :  {f1_n:.4f}")
    print(f"    Elevated  :  {f1_e:.4f}  {'<-- bottleneck' if f1_e < 0.55 else '(target met)'}")
    print(f"    Critical  :  {f1_c:.4f}")

    print(f"\n  Outputs")
    print(f"    Model   :  {results['model_path']}")
    print(f"    Report  :  {args.out_dir}/final_report.txt")
    print(f"    Plots   :  {RESULTS_DIR}/")

    print(f"\n  Total time: {elapsed:.0f}s")
    print(f"\n{'='*W}\n")


if __name__ == "__main__":
    main()
