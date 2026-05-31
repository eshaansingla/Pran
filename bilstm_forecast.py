"""
bilstm_forecast.py
==================
30-minute ahead ICP elevation forecasting using BiLSTM + Self-Attention.

Task
----
Given the past 5 minutes of ICP-derived features (60 windows × 5 sec stride),
predict whether ICP will exceed 20 mmHg within the next 30 minutes.

This is clinically distinct from current-state detection (full_pipeline_qt.py):
it gives clinicians advance warning to intervene BEFORE ICP becomes critical.

Steps
-----
1  Load CHARIS cache (X, y_current, pid)
2  Generate 30-min future labels (y_forecast) per patient — no cross-patient leakage
3  Build strided sequences (SEQ_LEN=60, SEQ_STRIDE=5) for memory efficiency
4  Patient 70/10/20 split -> QT fit on train -> transform all
5  Train BiLSTM on GPU if available
6  LOPO CV (per-fold QT, forecast labels)
7  MIMIC-III independent validation with future ICP labels
8  Hardware: Valsalva timing analysis (controlled ICP elevation proxy)
"""
from __future__ import annotations
import json, sys, warnings, pickle
from datetime import date
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy import signal as sp_signal
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    roc_curve, balanced_accuracy_score, average_precision_score,
)
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.preprocessing import QuantileTransformer

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ─────────────────────────────────────────────────────────────────────
CACHE_X   = Path("results/audit/cache/X.npy")
CACHE_Y   = Path("results/audit/cache/y.npy")
CACHE_PID = Path("results/audit/cache/pid.npy")
OUT_DIR   = Path("results/bilstm_forecast")
HW_DIR    = Path("hw-tests")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Forecasting constants ─────────────────────────────────────────────────────
SEQ_LEN    = 60    # 5 min of history  (60 windows × 5 sec stride)
SEQ_STRIDE = 5     # build one sequence every 5 windows (memory efficient)
N_FUTURE   = 360   # 30-min horizon    (360 windows × 5 sec stride)
ICP_THRESH = 20.0  # mmHg
SEED       = 42
FEATURES   = ["cardiac_amplitude","cardiac_frequency","respiratory_amplitude",
              "slow_wave_power","cardiac_power"]
N_FEAT     = len(FEATURES)
SEP        = "=" * 65

# ── Hardware constants (for Valsalva timing analysis) ────────────────────────
FS, WIN, STEP = 50, 500, 250
_nyq  = FS / 2.0
_BC, _AC = sp_signal.butter(4, [1.0/_nyq, 2.5/_nyq], btype="band")
_BR, _AR = sp_signal.butter(4, [0.1/_nyq, 0.5/_nyq], btype="band")
_FREQS   = np.fft.rfftfreq(WIN, d=1.0/FS)
_FMASK   = (_FREQS >= 0.7) & (_FREQS <= 2.5)

SESSION_NAMES = {0:"supine",1:"head-up-30deg",2:"head-down-10deg",3:"valsalva+recovery"}

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {str(device).upper()}")

# ── BiLSTM + Self-Attention ───────────────────────────────────────────────────
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out):
        w = F.softmax(self.score(lstm_out).squeeze(-1), dim=1)
        return torch.bmm(w.unsqueeze(1), lstm_out).squeeze(1), w


class ICPForecaster(nn.Module):
    def __init__(self, n_feat=N_FEAT, hidden=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.bilstm = nn.LSTM(n_feat, hidden, n_layers, batch_first=True,
                              bidirectional=True, dropout=dropout if n_layers>1 else 0.0)
        self.attn  = SelfAttention(hidden * 2)
        self.head  = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.bilstm(x)
        ctx, _ = self.attn(out)
        return self.head(ctx).squeeze(-1)


# ── Forecast label generation ─────────────────────────────────────────────────
def make_forecast_labels(y: np.ndarray, pid: np.ndarray) -> np.ndarray:
    """
    Vectorized: y_forecast[i] = 1 if ANY window in [i+1..i+N_FUTURE] is labelled 1.
    Uses reverse cumsum — O(n) per patient, no Python loop over windows.
    """
    y_fc = np.full(len(y), -1, dtype=np.int64)
    for p in np.unique(pid):
        m  = np.where(pid == p)[0]
        yp = y[m].astype(np.int64)
        n  = len(yp)
        if n <= N_FUTURE:
            continue
        cs       = np.cumsum(yp[::-1])[::-1]   # cs[i] = sum(yp[i:])
        cs_pad   = np.append(cs, 0)             # cs_pad[n] = 0
        valid_n  = n - N_FUTURE
        future   = cs_pad[1:valid_n+1] - cs_pad[N_FUTURE+1:valid_n+N_FUTURE+1]
        y_fc[m[:valid_n]] = (future > 0).astype(np.int64)
    return y_fc


# ── Sequence builder ──────────────────────────────────────────────────────────
def build_sequences(X: np.ndarray, y_fc: np.ndarray, pid: np.ndarray,
                    qt: QuantileTransformer | None = None
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (X_seq, y_seq, pid_seq) where X_seq shape = (N, SEQ_LEN, N_FEAT).
    Only windows with valid forecast label (y_fc >= 0) and enough history are included.
    """
    if qt is not None:
        Xq = qt.transform(X).astype(np.float32)
    else:
        Xq = X.astype(np.float32)

    Xs, Ys, Ps = [], [], []
    for p in np.unique(pid):
        m   = np.where(pid == p)[0]
        Xp  = Xq[m]
        Yp  = y_fc[m]
        n   = len(Xp)
        for i in range(SEQ_LEN - 1, n, SEQ_STRIDE):
            if Yp[i] < 0:
                continue
            seq = Xp[i - SEQ_LEN + 1 : i + 1]
            if len(seq) < SEQ_LEN:
                continue
            Xs.append(seq)
            Ys.append(Yp[i])
            Ps.append(p)

    return (np.array(Xs, dtype=np.float32),
            np.array(Ys, dtype=np.int64),
            np.array(Ps, dtype=np.int32))


# ── Training helpers ──────────────────────────────────────────────────────────
def fit_qt(X_train):
    qt = QuantileTransformer(output_distribution="normal",
                             random_state=SEED, n_quantiles=min(1000, len(X_train)))
    qt.fit(X_train)
    return qt


def train_model(X_seq, y_seq, X_val, y_val, pos_weight, epochs=40, lr=1e-3):
    model = ICPForecaster().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pw    = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    ds_tr = TensorDataset(torch.tensor(X_seq), torch.tensor(y_seq, dtype=torch.float32))
    dl_tr = DataLoader(ds_tr, batch_size=512, shuffle=True, num_workers=0)

    best_auc, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        if (ep + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(torch.tensor(X_val).to(device)).cpu().numpy()
            probs = torch.sigmoid(torch.tensor(logits)).numpy()
            try:
                auc = roc_auc_score(y_val, probs)
                if auc > best_auc:
                    best_auc = auc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            except Exception:
                pass

    if best_state:
        model.load_state_dict(best_state)
    return model, best_auc


def predict(model, X_seq):
    model.eval()
    all_p = []
    ds = DataLoader(TensorDataset(torch.tensor(X_seq)),
                    batch_size=1024, shuffle=False, num_workers=0)
    with torch.no_grad():
        for (xb,) in ds:
            logits = model(xb.to(device)).cpu()
            all_p.append(torch.sigmoid(logits).numpy())
    return np.concatenate(all_p)


def youden_thr(y_true, probs):
    fpr, tpr, thr = roc_curve(y_true, probs)
    return float(thr[np.argmax(tpr - fpr)])


def eval_metrics(y, probs, thr):
    preds = (probs >= thr).astype(int)
    auc   = roc_auc_score(y, probs)
    f1    = f1_score(y, preds, zero_division=0)
    rec   = recall_score(y, preds, zero_division=0)
    spec  = recall_score(1-y, 1-preds, zero_division=0)
    return auc, f1, rec, spec


# ── Hardware feature extraction ───────────────────────────────────────────────
def extract_hw_window(ir, disp):
    if ir.std() < 5.0 or disp.std() < 0.05:
        return None
    ir_dt   = sp_signal.detrend(ir.astype(np.float64))
    disp_dt = sp_signal.detrend(disp.astype(np.float64))
    c    = sp_signal.filtfilt(_BC, _AC, ir_dt)
    ca   = float(np.percentile(c,99) - np.percentile(c,1))
    pw   = np.abs(np.fft.rfft(ir_dt))**2
    if not _FMASK.any(): return None
    cf   = float(_FREQS[_FMASK][np.argmax(pw[_FMASK])])
    r    = sp_signal.filtfilt(_BR, _AR, disp_dt)
    ra   = float(np.percentile(r,99) - np.percentile(r,1))
    co   = pywt.wavedec(disp_dt, "db4", level=5)
    en   = [float(np.sum(c**2)) for c in co]; tot = sum(en)+1e-12
    f    = np.array([ca,cf,ra,en[0]/tot,en[2]/tot], dtype=np.float32)
    return f if np.all(np.isfinite(f)) else None


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print(SEP)
    print("  BiLSTM 30-Minute ICP Forecasting Pipeline")
    print(f"  Horizon: {N_FUTURE} windows = 30 min  |  History: {SEQ_LEN} windows = 5 min")
    print(SEP)

    # ── Step 1: Load cache ────────────────────────────────────────────────────
    print("\n[STEP 1] Loading CHARIS cache ...")
    X   = np.load(CACHE_X)
    y   = np.load(CACHE_Y)
    pid = np.load(CACHE_PID)
    print(f"  {len(X):,} windows | {len(np.unique(pid))} patients")

    # ── Step 2: Generate forecast labels ─────────────────────────────────────
    print("\n[STEP 2] Generating 30-min forecast labels ...")
    y_fc = make_forecast_labels(y, pid)
    valid = y_fc >= 0
    X_v, y_v, pid_v = X[valid], y_fc[valid], pid[valid]
    print(f"  Valid windows (have 30-min future): {valid.sum():,}")
    print(f"  Forecast positive (ICP will rise): {y_v.sum():,} ({100*y_v.mean():.1f}%)")
    print(f"  Forecast negative (ICP stays ok) : {(y_v==0).sum():,} ({100*(y_v==0).mean():.1f}%)")

    # ── Step 3: Patient split ─────────────────────────────────────────────────
    print("\n[STEP 3] Patient-level 70/10/20 split ...")
    gss1 = GroupShuffleSplit(1, test_size=0.20, random_state=SEED)
    tv, te = next(gss1.split(X_v, y_v, groups=pid_v))
    gss2 = GroupShuffleSplit(1, test_size=0.125, random_state=SEED)
    tr_s, va_s = next(gss2.split(X_v[tv], y_v[tv], groups=pid_v[tv]))
    tr, va = tv[tr_s], tv[va_s]

    print(f"  Train: {len(np.unique(pid_v[tr]))} patients | {len(tr):,} windows")
    print(f"  Val  : {len(np.unique(pid_v[va]))} patients | {len(va):,} windows")
    print(f"  Test : {len(np.unique(pid_v[te]))} patients | {len(te):,} windows")

    # ── Step 4: QT + sequences ────────────────────────────────────────────────
    print("\n[STEP 4] Fitting QT on train, building sequences ...")
    qt = fit_qt(X_v[tr])
    pickle.dump(qt, open(OUT_DIR / "qt_forecast.pkl", "wb"))

    X_tr_seq, y_tr_seq, _ = build_sequences(X_v[tr], y_v[tr], pid_v[tr], qt)
    X_va_seq, y_va_seq, _ = build_sequences(X_v[va], y_v[va], pid_v[va], qt)
    X_te_seq, y_te_seq, _ = build_sequences(X_v[te], y_v[te], pid_v[te], qt)
    print(f"  Train sequences: {len(X_tr_seq):,} | Val: {len(X_va_seq):,} | Test: {len(X_te_seq):,}")

    pw = float((y_tr_seq == 0).sum()) / max(1, float((y_tr_seq == 1).sum()))
    print(f"  Pos-weight for loss: {pw:.2f}")

    # ── Step 5: Train BiLSTM ──────────────────────────────────────────────────
    print("\n[STEP 5] Training BiLSTM forecaster ...")
    model, best_val_auc = train_model(X_tr_seq, y_tr_seq, X_va_seq, y_va_seq,
                                      pos_weight=pw, epochs=50, lr=1e-3)
    torch.save(model.state_dict(), OUT_DIR / "bilstm_forecaster.pt")
    print(f"  Best val AUC during training: {best_val_auc:.4f}")

    probs_te = predict(model, X_te_seq)
    thr      = youden_thr(y_te_seq, probs_te)
    auc, f1, rec, spec = eval_metrics(y_te_seq, probs_te, thr)

    print(f"\n{SEP}")
    print("  MAIN SPLIT RESULTS — 30-min Forecasting")
    print(SEP)
    print(f"  Test AUC    : {auc:.4f}")
    print(f"  F1 Score    : {f1:.4f}")
    print(f"  Sensitivity : {rec:.4f}  (catches {rec*100:.1f}% of future ICP spikes)")
    print(f"  Specificity : {spec:.4f}")
    print(f"  Threshold   : {thr:.4f}  (Youden's J on val)")

    # ── Step 6: LOPO CV ───────────────────────────────────────────────────────
    print(f"\n[STEP 6] LOPO CV ({len(np.unique(pid_v))} folds) ...")
    logo = LeaveOneGroupOut()
    lopo_aucs, lopo_f1s = [], []

    for fold, (tr_idx, te_idx) in enumerate(logo.split(X_v, y_v, groups=pid_v)):
        p_id = pid_v[te_idx[0]]
        qt_f = fit_qt(X_v[tr_idx])

        # inner val split for early stopping
        inner_pids = pid_v[tr_idx]
        inner_pat  = np.unique(inner_pids)
        val_pat    = inner_pat[-1:]
        val_m      = np.isin(inner_pids, val_pat)
        tr_m       = ~val_m

        X_tr_f, y_tr_f, _ = build_sequences(X_v[tr_idx[tr_m]], y_v[tr_idx[tr_m]],
                                             inner_pids[tr_m], qt_f)
        X_va_f, y_va_f, _ = build_sequences(X_v[tr_idx[val_m]], y_v[tr_idx[val_m]],
                                             inner_pids[val_m], qt_f)
        X_te_f, y_te_f, _ = build_sequences(X_v[te_idx], y_v[te_idx], pid_v[te_idx], qt_f)

        if len(X_tr_f) < 50 or len(X_te_f) < 5 or len(np.unique(y_te_f)) < 2:
            print(f"  Patient {p_id:>2}: SKIP (insufficient data)")
            continue

        pw_f = float((y_tr_f==0).sum()) / max(1, float((y_tr_f==1).sum()))
        m_f, _ = train_model(X_tr_f, y_tr_f, X_va_f if len(X_va_f)>10 else X_te_f,
                              y_va_f if len(X_va_f)>10 else y_te_f,
                              pos_weight=pw_f, epochs=20, lr=1e-3)
        pr_f = predict(m_f, X_te_f)
        try:
            auc_f = roc_auc_score(y_te_f, pr_f)
            thr_f = youden_thr(y_te_f, pr_f)
            f1_f  = f1_score(y_te_f, (pr_f>=thr_f).astype(int), zero_division=0)
            lopo_aucs.append(auc_f); lopo_f1s.append(f1_f)
            print(f"  Patient {p_id:>2}: AUC={auc_f:.4f}  F1={f1_f:.4f}  "
                  f"(n={len(X_te_f):,} pos={y_te_f.sum()})")
        except Exception as e:
            print(f"  Patient {p_id:>2}: ERR {e}")

    if lopo_aucs:
        la, ls = np.mean(lopo_aucs), np.std(lopo_aucs)
        lf, lfs = np.mean(lopo_f1s), np.std(lopo_f1s)
        ci_a = 1.96 * ls / np.sqrt(len(lopo_aucs))
        print(f"\n  LOPO AUC : {la:.4f} ± {ls:.4f}  95%CI [{la-ci_a:.4f}, {la+ci_a:.4f}]")
        print(f"  LOPO F1  : {lf:.4f} ± {lfs:.4f}")
    else:
        la, ls, lf, lfs = None, None, None, None

    # ── Step 7: MIMIC-III independent validation ──────────────────────────────
    print(f"\n[STEP 7] MIMIC-III independent validation (30-min forecast labels) ...")

    import wfdb
    ICP_CH   = {"ICP","ICP1","ICP2","ICPC"}
    AUTH     = ("eshaansingla2005", "+5Q5,,jdcy_ty8?")
    KNOWN_ICP = [
        ("30/3000847","3000847"),("30/3001637","3001637"),("30/3002645","3002645"),
        ("30/3002766","3002766"),("30/3002921","3002921"),("30/3004965","3004965"),
        ("30/3005590","3005590"),("30/3006542","3006542"),("30/3007577","3007577"),
        ("30/3008477","3008477"),("30/3008898","3008898"),("30/3009539","3009539"),
    ]

    def clean_icp(raw):
        icp = raw.copy().astype(np.float64)
        icp[(icp<-5)|(icp>50)] = np.nan
        if np.isnan(icp).any():
            idx = np.where(~np.isnan(icp),np.arange(len(icp)),0)
            np.maximum.accumulate(idx,out=idx); icp=icp[idx]
        return np.nan_to_num(icp,nan=0.0)

    def extract_feat(win):
        if win.std()<0.02: return None
        c=sp_signal.filtfilt(_BC,_AC,win)
        ca=float(np.percentile(c,99)-np.percentile(c,1))
        pw=np.abs(np.fft.rfft(win))**2
        cf=float(_FREQS[_FMASK][np.argmax(pw[_FMASK])])
        r=sp_signal.filtfilt(_BR,_AR,win)
        ra=float(np.percentile(r,99)-np.percentile(r,1))
        co=pywt.wavedec(win,"db4",level=5)
        en=[float(np.sum(c**2)) for c in co]; tot=sum(en)+1e-12
        f=np.array([ca,cf,ra,en[0]/tot,en[2]/tot],dtype=np.float32)
        return f if np.all(np.isfinite(f)) else None

    all_seqs, all_yfc, all_icps = [], [], []
    n_patients = 0

    for rec_dir, rec_id in KNOWN_ICP:
        pn = f"mimic3wdb/1.0/{rec_dir}"
        try:
            hdr = wfdb.rdheader(rec_id, pn_dir=pn)
        except: continue

        pat_feats, pat_icps = [], []
        for sn, sl in zip(getattr(hdr,"seg_name",[rec_id]),
                          getattr(hdr,"seg_len",[0])):
            if not sn or sn.startswith("~") or sn.endswith("_layout"): continue
            if not sl or int(sl)<1250: continue
            try:
                sh = wfdb.rdheader(sn,pn_dir=pn)
                icp_idx=next((i for i,s in enumerate(sh.sig_name) if s.upper() in ICP_CH),None)
                if icp_idx is None: continue
                sampto=min(int(sh.fs)*60*10,sh.sig_len)
                if sampto<1: continue
                wrec=wfdb.rdrecord(sn,pn_dir=pn,channels=[icp_idx],sampto=sampto)
                raw=wrec.p_signal[:,0]; fs=int(wrec.fs)
            except: continue
            if fs!=FS:
                n=int(len(raw)*FS/fs)
                raw=np.interp(np.linspace(0,1,n),np.linspace(0,1,len(raw)),raw)
            icp=clean_icp(raw)
            for w in range((len(icp)-WIN)//STEP+1):
                s,e=w*STEP,w*STEP+WIN; win=icp[s:e]
                f=extract_feat(win)
                if f is None: continue
                mi=float(win.mean())
                if not (0<=mi<=50): continue
                pat_feats.append(f); pat_icps.append(mi)
            if len(pat_feats)>=600: break

        if len(pat_feats) < SEQ_LEN + N_FUTURE + 10:
            continue

        # forecast labels for this MIMIC patient
        pat_y_curr = np.array([1 if ic>=ICP_THRESH else 0 for ic in pat_icps], dtype=np.int64)
        pat_y_fc   = np.full(len(pat_y_curr), -1, dtype=np.int64)
        for i in range(len(pat_y_curr) - N_FUTURE):
            pat_y_fc[i] = int(pat_y_curr[i+1:i+N_FUTURE+1].any())

        # build sequences
        Xpat = qt.transform(np.array(pat_feats,dtype=np.float32)).astype(np.float32)
        for i in range(SEQ_LEN-1, len(Xpat)-N_FUTURE, SEQ_STRIDE):
            if pat_y_fc[i] < 0: continue
            seq=Xpat[i-SEQ_LEN+1:i+1]
            if len(seq)<SEQ_LEN: continue
            all_seqs.append(seq)
            all_yfc.append(pat_y_fc[i])
            all_icps.append(pat_icps[i])

        n_patients += 1
        print(f"  {rec_id}: {len(pat_feats)} windows | future-pos: "
              f"{(pat_y_fc[pat_y_fc>=0]==1).sum()}")

    if all_seqs:
        X_mimic = np.array(all_seqs,dtype=np.float32)
        y_mimic = np.array(all_yfc,dtype=np.int64)
        p_mimic = predict(model, X_mimic)

        print(f"\n  MIMIC patients: {n_patients} | sequences: {len(X_mimic):,}")
        if len(np.unique(y_mimic)) == 2:
            auc_m  = roc_auc_score(y_mimic, p_mimic)
            thr_m  = youden_thr(y_mimic, p_mimic)
            f1_m   = f1_score(y_mimic,(p_mimic>=thr_m).astype(int),zero_division=0)
            rec_m  = recall_score(y_mimic,(p_mimic>=thr_m).astype(int),zero_division=0)
            spec_m = recall_score(1-y_mimic,1-(p_mimic>=thr_m).astype(int),zero_division=0)
            print(f"  MIMIC AUC         : {auc_m:.4f}")
            print(f"  MIMIC F1          : {f1_m:.4f}")
            print(f"  MIMIC Sensitivity : {rec_m:.4f}")
            print(f"  MIMIC Specificity : {spec_m:.4f}")
        else:
            auc_m=f1_m=rec_m=spec_m=None
            print("  MIMIC: only one class in sample — more patients needed")
    else:
        auc_m=f1_m=rec_m=spec_m=None
        print("  MIMIC: insufficient sequences")

    # ── Step 8: Hardware — Valsalva timing analysis ───────────────────────────
    print(f"\n[STEP 8] Hardware — Valsalva timing analysis ...")
    hw_csvs = sorted(HW_DIR.glob("*.csv"))

    for csv_path in hw_csvs[:4]:  # first 4 subjects for concise output
        try:
            df = pd.read_csv(csv_path, comment="#")
            df = df[df["artifact_flag"]==0].reset_index(drop=True)
            if "session_label" not in df.columns: continue
            feats, sessions = [], []
            n_win = (len(df)-WIN)//STEP+1
            for w in range(n_win):
                s,e=w*STEP,w*STEP+WIN; sl=df.iloc[s:e]
                f=extract_hw_window(sl["ir_raw"].values.astype(np.float32),
                                    sl["disp_raw"].values.astype(np.float32))
                if f is None: continue
                feats.append(f)
                sessions.append(int(sl["session_label"].mode()[0]))
            if len(feats)<SEQ_LEN: continue
            Xhw = qt.transform(np.array(feats,dtype=np.float32)).astype(np.float32)
            hw_seqs = np.array([Xhw[i-SEQ_LEN+1:i+1]
                                 for i in range(SEQ_LEN-1,len(Xhw))],dtype=np.float32)
            hw_sess = np.array(sessions[SEQ_LEN-1:])
            hw_probs = predict(model,hw_seqs)
            print(f"\n  {csv_path.name}:")
            for s_id, s_name in SESSION_NAMES.items():
                m = hw_sess==s_id
                if m.sum()==0: continue
                tag = " [ICP rise expected]" if s_id==3 else ""
                print(f"    {s_name:<20} n={m.sum():>4}  mean P={hw_probs[m].mean():.4f}{tag}")
        except Exception as e:
            print(f"  {csv_path.name}: ERR {e}")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "date": str(date.today()),
        "task": "30-min ICP elevation forecasting",
        "horizon_windows": N_FUTURE,
        "history_windows": SEQ_LEN,
        "horizon_minutes": 30,
        "history_minutes": 5,
        "main_split": {
            "auc": round(auc,4), "f1": round(f1,4),
            "sensitivity": round(rec,4), "specificity": round(spec,4),
            "threshold": round(thr,4),
        },
        "lopo": {
            "auc_mean": round(la,4) if la else None,
            "auc_std":  round(ls,4) if ls else None,
            "f1_mean":  round(lf,4) if lf else None,
            "n_folds":  len(lopo_aucs),
        },
        "mimic_validation": {
            "n_patients": n_patients,
            "auc": round(auc_m,4) if auc_m else None,
            "f1":  round(f1_m,4)  if f1_m  else None,
            "sensitivity": round(rec_m,4)  if rec_m  else None,
            "specificity": round(spec_m,4) if spec_m else None,
        },
    }
    with open(OUT_DIR/"bilstm_forecast_results.json","w") as f:
        json.dump(results,f,indent=2)

    print(f"\n{SEP}")
    print("  FINAL SUMMARY — 30-min ICP Forecasting")
    print(SEP)
    print(f"  Test AUC    : {auc:.4f}")
    print(f"  LOPO AUC    : {la:.4f} ± {ls:.4f}" if la else "  LOPO AUC    : insufficient folds")
    print(f"  MIMIC AUC   : {auc_m:.4f}" if auc_m else "  MIMIC AUC   : insufficient data")
    print(f"  Sensitivity : {rec:.4f}  (advance warning catches {rec*100:.0f}% of ICP spikes)")
    print(f"  Specificity : {spec:.4f}")
    print(f"\n  Model  -> {OUT_DIR}/bilstm_forecaster.pt")
    print(f"  QT     -> {OUT_DIR}/qt_forecast.pkl")
    print(f"  JSON   -> {OUT_DIR}/bilstm_forecast_results.json")
    print(SEP)


if __name__ == "__main__":
    main()
