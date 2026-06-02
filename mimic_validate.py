"""
mimic_validate.py  —  MIMIC-III independent invasive validation
2 patients: 1 clearly normal ICP, 1 clearly elevated ICP
Features extracted from ICP waveform at 50 Hz — same domain as CHARIS training
"""
import sys, pickle, warnings
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path
import numpy as np
import wfdb
import xgboost as xgb
import pywt
from scipy import signal as sp_signal
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

# ── constants (must match regen_cache.py exactly) ─────────────────────────────
FS, WIN, STEP = 50, 500, 250
ICP_CH  = {"ICP", "ICP1", "ICP2", "ICPC"}
THRESH  = 20.0
FEATS   = ["cardiac_amplitude","cardiac_frequency","respiratory_amplitude",
           "slow_wave_power","cardiac_power"]
_nyq    = FS / 2.0
_BC, _AC = sp_signal.butter(4, [1.0/_nyq, 2.5/_nyq], btype="band")
_BR, _AR = sp_signal.butter(4, [0.1/_nyq, 0.5/_nyq], btype="band")
_FREQS   = np.fft.rfftfreq(WIN, d=1.0/FS)
_FMASK   = (_FREQS >= 0.7) & (_FREQS <= 2.5)

CREDS  = ("eshaansingla2005", "+5Q5,,jdcy_ty8?")
SEP    = "=" * 60

# ── load model ────────────────────────────────────────────────────────────────
print(SEP)
print("  MIMIC-III Independent Validation")
print(SEP)

assert Path("models/xgb_qt.json").exists(),  "models/xgb_qt.json not found"
assert Path("models/qt_scaler.pkl").exists(), "models/qt_scaler.pkl not found"
bst = xgb.Booster()
bst.load_model("models/xgb_qt.json")
qt  = pickle.load(open("models/qt_scaler.pkl", "rb"))
print("  Model loaded OK")

# ── feature extraction ────────────────────────────────────────────────────────
def extract(win):
    if win.std() < 0.02: return None
    c = sp_signal.filtfilt(_BC, _AC, win)
    ca = float(np.percentile(c,99) - np.percentile(c,1))
    pw = np.abs(np.fft.rfft(win))**2
    cf = float(_FREQS[_FMASK][np.argmax(pw[_FMASK])])
    r  = sp_signal.filtfilt(_BR, _AR, win)
    ra = float(np.percentile(r,99) - np.percentile(r,1))
    co = pywt.wavedec(win, "db4", level=5)
    en = [float(np.sum(c**2)) for c in co]; tot = sum(en)+1e-12
    f  = np.array([ca, cf, ra, en[0]/tot, en[2]/tot], dtype=np.float32)
    return f if np.all(np.isfinite(f)) else None

def clean_icp(raw):
    icp = raw.copy().astype(np.float64)
    icp[(icp < -5) | (icp > 50)] = np.nan
    if np.isnan(icp).any():
        idx = np.where(~np.isnan(icp), np.arange(len(icp)), 0)
        np.maximum.accumulate(idx, out=idx)
        icp = icp[idx]
    return np.nan_to_num(icp, nan=0.0)

def get_windows(rec_id, pn_dir):
    """Return (features, mean_icp_per_window) from all valid segments."""
    feats, icps = [], []
    try:
        hdr = wfdb.rdheader(rec_id, pn_dir=pn_dir)
    except Exception as e:
        print(f"    header error: {e}"); return feats, icps

    seg_names = getattr(hdr, "seg_name", [rec_id])
    seg_lens  = getattr(hdr, "seg_len",  [hdr.sig_len if hasattr(hdr,"sig_len") else 0])

    for sn, sl in zip(seg_names, seg_lens):
        if not sn or sn.startswith("~") or sn.endswith("_layout"): continue
        if not sl or int(sl) < 1250: continue          # need ≥10s at 125Hz

        try:
            sh = wfdb.rdheader(sn, pn_dir=pn_dir)
            sig_upper = [s.upper() for s in sh.sig_name]
            icp_idx = next((i for i,s in enumerate(sig_upper) if s in ICP_CH), None)
            if icp_idx is None: continue

            sampto = min(int(sh.fs) * 60 * 10, sh.sig_len)  # cap 10 min, never exceed sig_len
            if sampto < 1: continue

            rec = wfdb.rdrecord(sn, pn_dir=pn_dir, channels=[icp_idx], sampto=sampto)
            raw = rec.p_signal[:, 0]
            fs  = int(rec.fs)
        except Exception as e:
            continue

        # resample to 50 Hz
        if fs != FS:
            n = int(len(raw) * FS / fs)
            raw = np.interp(np.linspace(0,1,n), np.linspace(0,1,len(raw)), raw)

        icp = clean_icp(raw)

        for w in range((len(icp) - WIN) // STEP + 1):
            s, e = w*STEP, w*STEP+WIN
            win  = icp[s:e]
            f    = extract(win)
            if f is None: continue
            mi   = float(win.mean())
            if not (0 <= mi <= 50): continue
            feats.append(f); icps.append(mi)

        if len(feats) >= 300: break   # enough windows from this patient

    return feats, icps

# ── known ICP records from MIMIC-III (confirmed from earlier scan) ─────────────
# These all showed ICP channel in layout header during earlier run
KNOWN_ICP = [
    ("30/3000847", "3000847"),
    ("30/3001637", "3001637"),
    ("30/3002094", "3002094"),
    ("30/3002645", "3002645"),
    ("30/3002766", "3002766"),
    ("30/3002921", "3002921"),
    ("30/3004965", "3004965"),
    ("30/3005590", "3005590"),
    ("30/3006542", "3006542"),
    ("30/3007577", "3007577"),
    ("30/3008006", "3008006"),
    ("30/3008477", "3008477"),
    ("30/3008823", "3008823"),
    ("30/3008898", "3008898"),
    ("30/3009539", "3009539"),
    ("30/3009722", "3009722"),
]

print("\n  Extracting windows from known ICP patients ...")
all_feats, all_icps, all_pid = [], [], []
found = 0

for rec_dir, rec_id in KNOWN_ICP:
    pn_dir = f"mimic3wdb/1.0/{rec_dir}"
    print(f"\n  [{found+1}] {rec_id} — extracting ...", flush=True)
    f, ic = get_windows(rec_id, pn_dir)
    print(f"      → {len(f)} windows | ICP range: "
          f"{min(ic):.1f}–{max(ic):.1f} mmHg | mean: {np.mean(ic):.1f} mmHg" if f
          else "      → 0 windows (skip)")
    if len(f) < 10: continue
    all_feats.extend(f); all_icps.extend(ic)
    all_pid.extend([found+1]*len(f))
    found += 1
    if found >= 12: break

if not all_feats:
    print("\nERROR: No windows extracted from any patient."); sys.exit(1)

# ── predict ───────────────────────────────────────────────────────────────────
X    = np.array(all_feats, dtype=np.float32)
IC   = np.array(all_icps,  dtype=np.float32)
PID  = np.array(all_pid,   dtype=np.int32)
X_qt = qt.transform(X).astype(np.float32)
P    = bst.predict(xgb.DMatrix(X_qt, feature_names=FEATS))

print(f"\n{SEP}")
print("  RESULTS")
print(SEP)
print(f"\n  Total windows : {len(X):,}  |  Patients: {found}")

r_p,  pv_p  = pearsonr(IC, P)
r_sp, pv_sp = spearmanr(IC, P)
low  = P[IC <  THRESH]; high = P[IC >= THRESH]
y_bin = (IC >= THRESH).astype(int)

# ── Classification metrics ─────────────────────────────────────────────────
_thr_path = Path("models/xgb_qt_thr.pkl")
if _thr_path.exists():
    import pickle as _pkl
    _thr_val, _ = _pkl.load(open(_thr_path, "rb"))
else:
    _thr_val = 0.5
pred_bin = (P >= _thr_val).astype(int)

mimic_auc = roc_auc_score(y_bin, P) if y_bin.sum() > 0 and (1-y_bin).sum() > 0 else float("nan")
mimic_acc = float((pred_bin == y_bin).mean())
mimic_f1  = f1_score(y_bin, pred_bin, zero_division=0)
cm_m      = confusion_matrix(y_bin, pred_bin)

print(f"\n  Pearson r     : {r_p:+.4f}  (p={pv_p:.2e})")
print(f"  Spearman rho  : {r_sp:+.4f}  (p={pv_sp:.2e})")
print(f"\n  AUC           : {mimic_auc:.4f}")
print(f"  Accuracy      : {mimic_acc:.4f}  (threshold={_thr_val:.4f})")
print(f"  F1            : {mimic_f1:.4f}")
print(f"\n  Mean P | ICP <  20 mmHg : {low.mean():.4f}  (n={len(low):,})")
print(f"  Mean P | ICP >= 20 mmHg : {high.mean():.4f}  (n={len(high):,})")
print(f"  Separation : {high.mean()-low.mean():+.4f}  "
      f"({'CORRECT direction' if high.mean()>low.mean() else 'WRONG direction'})")

print(f"\n  Per-patient:")
print(f"  {'ID':>4}  {'N':>5}  {'MeanICP':>8}  {'Elev%':>7}  {'MeanP':>7}  {'r':>7}")
print(f"  {'─'*48}")
for pid in np.unique(PID):
    m = PID==pid; ic_p=IC[m]; pr_p=P[m]
    rr = pearsonr(ic_p,pr_p)[0] if len(ic_p)>5 else float("nan")
    print(f"  {pid:>4}  {m.sum():>5}  {ic_p.mean():>8.2f}  "
          f"{100*(ic_p>=THRESH).mean():>6.1f}%  {pr_p.mean():>7.4f}  {rr:>+7.4f}")

print(f"\n{SEP}")
