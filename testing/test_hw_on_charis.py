"""TEST B: hardware-trained XGBoost scored on CHARIS (ICP >= 20 mmHg label, 13 brain-injury patients, invasive ICP probe).
The saved hardware model needs 32 optical features (IR / displacement / red); CHARIS only has one ICP waveform, so it can only supply the
5 base features. So we use the hardware model recipe restricted to those 5 columns (same data, split, seed, hyper-parameters, QuantileTransformer,
early stopping + Youden threshold on separate validation subjects), trained on hardware subjects only, then applied to CHARIS untouched.
CHARIS features are computed with the hardware extractor (pran.features._base_features: detrended, amplitudes relative to window std), with the ICP
waveform used as both the 'ir' and 'displacement' channel. Every 4th window (step 1000 samples) is used for speed.
Run from repo root:  python testing/test_hw_on_charis.py
"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, wfdb
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "hardware-vasalva-method")); import common as C   # chdir's to repo root
from pran.features import _base_features, WIN, STEP
from sklearn.metrics import roc_auc_score

CACHE = OUT / "charis_hwstyle_features.npz"
if not CACHE.exists():
    Xa, ya, pa = [], [], []
    for h in sorted((ROOT / "data/raw/charis").glob("*.hea"), key=lambda f: int("".join(filter(str.isdigit, f.stem)))):
        pid = int("".join(filter(str.isdigit, h.stem))); rec = wfdb.rdrecord(str(h.with_suffix("")))
        sig = [s.upper() for s in rec.sig_name]; ii = next((i for i, s in enumerate(sig) if s in {"ICP", "ICP1", "ICP2", "ICPC"}), None)
        if ii is None: continue
        icp = rec.p_signal[:, ii].astype(np.float64)
        if int(rec.fs) != 50: n = int(len(icp) * 50 / int(rec.fs)); icp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(icp)), icp)
        icp[(icp < -5) | (icp > 50)] = np.nan; nan = np.isnan(icp)
        if nan.any(): idx = np.where(~nan, np.arange(len(icp)), 0); np.maximum.accumulate(idx, out=idx); icp = icp[idx]
        icp = np.nan_to_num(icp, nan=0.0); k = 0
        for w in range(0, (len(icp) - WIN) // STEP + 1, 4):
            win = icp[w * STEP:w * STEP + WIN]
            if win.std() < 0.02: continue
            f = _base_features(win, win)
            if f is None: continue
            Xa.append(f); ya.append(int((win >= 20).mean() > 0.60)); pa.append(pid); k += 1
        print("charis", pid, k, flush=True)
    np.savez(CACHE, X=np.array(Xa), y=np.array(ya), pid=np.array(pa))
z = np.load(CACHE); Xc, yc, pc = z["X"], z["y"], z["pid"]

# ---- train the 5-feature hardware model on hardware subjects only (same protocol as common.py)
X, y, P, S, age, names = C.load(); dev, test = C.split_subjects(P, age); cols = slice(0, 5)
va_s = np.random.default_rng(0).choice(dev, C.N_VAL, replace=False); va = np.where(np.isin(P, va_s))[0]; fit = np.where(np.isin(P, np.setdiff1d(dev, va_s)))[0]; te = np.where(np.isin(P, test))[0]
m, qt, thr, it = C.fit_scaled("XGBoost", X, y, fit, va, cols=cols)
s_hw = C.prob(m, qt, X[te], cols); hw = C.metrics(y[te], s_hw, thr)
print(f"hardware 5-feature model on its own held-out hardware subjects: AUC {hw['auc']:.3f}")

# ---- apply to CHARIS
s = C.prob(m, qt, Xc); rows = []
for p in np.unique(pc):
    k = pc == p
    if 0 < yc[k].sum() < k.sum(): rows.append(dict(patient=int(p), windows=int(k.sum()), pct_elevated=100 * yc[k].mean(), auc=roc_auc_score(yc[k], s[k]), mean_score=s[k].mean()))
df = pd.DataFrame(rows); df.to_csv(OUT / "B_hw_model_on_charis_per_patient.csv", index=False)
mm = C.metrics(yc, s, thr); pm = df.groupby("patient").mean_score.first()
res = dict(model="XGBoost, 5 base features, trained on hardware Valsalva-vs-rest (subjects only)", hw_threshold=thr, boosting_rounds=it,
           hw_own_heldout_auc=hw["auc"], charis_windows=int(len(yc)), charis_patients=int(len(np.unique(pc))), charis_pct_elevated=float(100 * yc.mean()),
           pooled_auc=mm["auc"], mean_per_patient_auc=float(df.auc.mean()), median_per_patient_auc=float(df.auc.median()), worst_patient_auc=float(df.auc.min()),
           patients_auc_above_0_5=f"{int((df.auc > .5).sum())}/{len(df)}", patients_auc_below_0_5=int((df.auc < .5).sum()),
           sensitivity=mm["sensitivity"], specificity=mm["specificity"], balanced_accuracy=mm["balanced_accuracy"])
(OUT / "B_hw_model_on_charis.json").write_text(json.dumps(res, indent=1, default=float)); print(json.dumps(res, indent=1, default=float)); print(df.round(3).to_string(index=False))
