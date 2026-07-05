"""
bilstm_classify.py
==================
BiLSTM current-state ICP elevation classifier.
Same task as hybrid_pipeline_qt.py (is ICP elevated right now?),
but uses sequences of windows for temporal context.

Same hybrid training strategy:
  - CHARIS abnormal sequences (subsampled to 5:1 cap)
  - HW all sequences (normals + valsalva), labeled by session
  - LOPO over HW patients — each patient's fold uses unbiased test AUC

Architecture: BiLSTM(64, 2-layer) → SelfAttention → LayerNorm → Dense(1)

Key differences vs XGBoost:
  - Input: 10 consecutive windows (50-second context) instead of single window
  - Sequences built WITHIN sessions (no cross-session contamination)
  - pos_weight in BCEWithLogitsLoss (no SMOTE on sequences — interpolating
    between temporal sequences is physiologically invalid)
  - Inner val patient for early stopping per LOPO fold

Run
---
    cd "C:\\Users\\asus\\Documents\\GitHub\\Pran"
    python bilstm_classify.py
"""
from __future__ import annotations
import json, pickle, sys, warnings
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
from scipy.stats import wilcoxon as _wilcoxon
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
    confusion_matrix,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import QuantileTransformer

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ──────────────────────────────────────────────────────────────────────
CACHE_X   = Path("results/audit/cache/X.npy")
CACHE_Y   = Path("results/audit/cache/y.npy")
CACHE_PID = Path("results/audit/cache/pid.npy")
HW_DIR    = Path("hw-tests")
OUT_DIR   = Path("results/bilstm_classify")
MODEL_DIR = Path("models/bilstm")
LOPO_DIR  = MODEL_DIR / "lopo"

# ── Hyperparameters ────────────────────────────────────────────────────────────
SEQ_LEN    = 10     # windows per sequence  (10 × 5-sec stride = 50-sec context)
SEQ_STRIDE = 2      # new sequence every 2 windows (10 seconds)
HIDDEN     = 64
N_LAYERS   = 2
DROPOUT    = 0.35
EPOCHS_LOPO  = 35
EPOCHS_FINAL = 50
BATCH      = 256
LR         = 1e-3
MAX_RATIO  = 5.0    # CHARIS abn cap: pos/neg <= 5 before pos_weight

FEATURES = ["cardiac_amplitude", "cardiac_frequency", "respiratory_amplitude",
            "slow_wave_power", "cardiac_power"]
N_FEAT   = len(FEATURES)
SEED     = 42

FS, WIN, STEP = 50, 500, 250
_nyq             = FS / 2.0
_B_CARD, _A_CARD = sp_signal.butter(4, [1.0/_nyq, 2.5/_nyq], btype="band")
_B_RESP, _A_RESP = sp_signal.butter(4, [0.1/_nyq, 0.5/_nyq], btype="band")
_FREQS           = np.fft.rfftfreq(WIN, d=1.0 / FS)
_FREQ_MASK       = (_FREQS >= 0.7) & (_FREQS <= 2.5)

torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model ──────────────────────────────────────────────────────────────────────
class SelfAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1, bias=False)

    def forward(self, x):
        w = F.softmax(self.score(x).squeeze(-1), dim=1)      # (B, T)
        return (w.unsqueeze(-1) * x).sum(dim=1), w            # (B, dim), (B, T)


class ICPClassifier(nn.Module):
    def __init__(self, n_feat=N_FEAT, hidden=HIDDEN, n_layers=N_LAYERS,
                 dropout=DROPOUT):
        super().__init__()
        self.bilstm = nn.LSTM(n_feat, hidden, n_layers, batch_first=True,
                              bidirectional=True,
                              dropout=dropout if n_layers > 1 else 0.0)
        self.attn = SelfAttention(hidden * 2)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1),
        )

    def forward(self, x):                    # x: (B, T, F)
        out, _ = self.bilstm(x)              # (B, T, 2H)
        ctx, _ = self.attn(out)              # (B, 2H)
        return self.head(ctx).squeeze(-1)    # (B,) logits


# ── Feature extraction ─────────────────────────────────────────────────────────
def extract_hw_window(ir: np.ndarray, disp: np.ndarray) -> np.ndarray | None:
    if ir.std() < 5.0 or disp.std() < 0.05:
        return None
    ir_dt   = sp_signal.detrend(ir.astype(np.float64))
    disp_dt = sp_signal.detrend(disp.astype(np.float64))

    c        = sp_signal.filtfilt(_B_CARD, _A_CARD, ir_dt)
    card_amp = float(np.percentile(c, 99) - np.percentile(c, 1))

    pwr = np.abs(np.fft.rfft(ir_dt)) ** 2
    if not _FREQ_MASK.any(): return None
    card_freq = float(_FREQS[_FREQ_MASK][np.argmax(pwr[_FREQ_MASK])])

    r        = sp_signal.filtfilt(_B_RESP, _A_RESP, disp_dt)
    resp_amp = float(np.percentile(r, 99) - np.percentile(r, 1))

    coeffs   = pywt.wavedec(disp_dt, "db4", level=5)
    energies = [float(np.sum(c ** 2)) for c in coeffs]
    total    = sum(energies) + 1e-12
    slow_pow    = energies[0] / total
    cardiac_pow = energies[2] / total

    feat = np.array([card_amp, card_freq, resp_amp, slow_pow, cardiac_pow],
                    dtype=np.float32)
    return feat if np.all(np.isfinite(feat)) else None


# ── Hardware loader ────────────────────────────────────────────────────────────
def load_hw_labeled(hw_dir: Path):
    """Returns (X, y, pid, sessions, names). sessions = per-window session_label."""
    X_all, y_all, pid_all, sess_all = [], [], [], []
    names: list[str] = []
    pid_idx = 0

    for csv_path in sorted(hw_dir.glob("*.csv")):
        df = pd.read_csv(csv_path, comment="#")
        required = {"ir_raw", "disp_raw", "artifact_flag", "session_label"}
        if not required.issubset(df.columns):
            continue
        df = df[df["artifact_flag"] == 0].reset_index(drop=True)
        n_win = (len(df) - WIN) // STEP + 1
        n_added = 0
        for w in range(n_win):
            s, e = w * STEP, w * STEP + WIN
            sl   = df.iloc[s:e]
            feat = extract_hw_window(sl["ir_raw"].values.astype(np.float32),
                                     sl["disp_raw"].values.astype(np.float32))
            if feat is None: continue
            sess  = int(sl["session_label"].mode()[0])
            X_all.append(feat); y_all.append(1 if sess == 3 else 0)
            pid_all.append(pid_idx); sess_all.append(sess)
            n_added += 1
        if n_added > 0:
            names.append(csv_path.name)
            pid_idx += 1

    return (np.array(X_all,    dtype=np.float32),
            np.array(y_all,    dtype=np.int32),
            np.array(pid_all,  dtype=np.int32),
            np.array(sess_all, dtype=np.int32),
            names)


# ── Sequence builders ──────────────────────────────────────────────────────────
def build_hw_sequences(hw_X_qt, hw_y, hw_pid, hw_sess,
                       seq_len=SEQ_LEN, stride=SEQ_STRIDE):
    """
    Build sequences WITHIN each patient×session to avoid cross-session contamination.
    Returns (X_seq, y_seq, pid_seq).
    """
    Xs, Ys, Ps = [], [], []
    for p in np.unique(hw_pid):
        for s in [0, 1, 2, 3]:
            m = (hw_pid == p) & (hw_sess == s)
            if m.sum() < seq_len:
                continue
            Xps = hw_X_qt[m]
            label = 1 if s == 3 else 0
            n = len(Xps)
            for i in range(seq_len - 1, n, stride):
                seq = Xps[i - seq_len + 1 : i + 1]
                if len(seq) < seq_len: continue
                Xs.append(seq); Ys.append(label); Ps.append(p)
    return (np.array(Xs, dtype=np.float32),
            np.array(Ys, dtype=np.int32),
            np.array(Ps, dtype=np.int32))


def build_charis_sequences(charis_X_qt, charis_y, charis_pid, abnormal_only=True,
                           seq_len=SEQ_LEN, stride=SEQ_STRIDE):
    """
    Build sequences within each CHARIS patient.
    If abnormal_only=True, keep only sequences whose LAST window is y=1.
    Assumes windows are stored in temporal order within each patient.
    """
    Xs, Ys, Ps = [], [], []
    for p in np.unique(charis_pid):
        m  = charis_pid == p
        Xp = charis_X_qt[m]
        Yp = charis_y[m]
        n  = len(Xp)
        for i in range(seq_len - 1, n, stride):
            if abnormal_only and Yp[i] != 1: continue
            seq = Xp[i - seq_len + 1 : i + 1]
            if len(seq) < seq_len: continue
            Xs.append(seq); Ys.append(int(Yp[i])); Ps.append(p)
    return (np.array(Xs, dtype=np.float32),
            np.array(Ys, dtype=np.int32),
            np.array(Ps, dtype=np.int32))


def subsample_sequences(X_seq, y_seq, pid_seq, n_target: int, seed: int = SEED):
    """Randomly subsample to n_target sequences."""
    if len(X_seq) <= n_target:
        return X_seq, y_seq, pid_seq
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_seq), n_target, replace=False)
    return X_seq[idx], y_seq[idx], pid_seq[idx]


# ── Utilities ──────────────────────────────────────────────────────────────────
def fit_qt(X: np.ndarray) -> QuantileTransformer:
    qt = QuantileTransformer(output_distribution="normal",
                             random_state=SEED, n_quantiles=min(1000, len(X)))
    return qt.fit(X)


def youden_threshold(y_true, probs) -> float:
    fpr, tpr, thr = roc_curve(y_true, probs)
    return float(thr[np.argmax(tpr - fpr)])


def pos_weight_tensor(y: np.ndarray) -> torch.Tensor:
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    pw = n0 / max(n1, 1)
    return torch.tensor([pw], dtype=torch.float32)


# ── Training helpers ───────────────────────────────────────────────────────────
def train_model(X_tr, y_tr, X_val, y_val, epochs: int, seed: int = SEED):
    """Train ICPClassifier with early stopping on val AUC."""
    torch.manual_seed(seed)
    model   = ICPClassifier().to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pw      = pos_weight_tensor(y_tr).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    ds_tr = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.float32))
    dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, num_workers=0,
                       pin_memory=DEVICE.type == "cuda")

    best_auc, best_state, patience = 0.0, None, 0
    MAX_PATIENCE = 8  # check every 2 epochs → 16 epochs of no improvement

    for ep in range(epochs):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        if (ep + 1) % 2 == 0 and len(X_val) > 0:
            probs = _predict_probs(model, X_val)
            try:
                auc = roc_auc_score(y_val, probs)
                if auc > best_auc:
                    best_auc = auc; patience = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience += 1
            except Exception:
                pass
            if patience >= MAX_PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_auc


def _predict_probs(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    all_p = []
    ds = DataLoader(TensorDataset(torch.tensor(X)),
                    batch_size=1024, shuffle=False, num_workers=0)
    with torch.no_grad():
        for (xb,) in ds:
            logits = model(xb.to(DEVICE)).cpu()
            all_p.append(torch.sigmoid(logits).numpy())
    return np.concatenate(all_p)


# ── LOPO ───────────────────────────────────────────────────────────────────────
def run_lopo(hw_X, hw_y, hw_pid, hw_sess, charis_X, charis_y, charis_pid,
             names, use_cache=True):
    """
    LOPO over HW patients.  CHARIS abnormal seqs in train (subsampled to 5:1).
    Inner val = last HW patient in train fold (for early stopping + threshold).
    """
    LOPO_DIR.mkdir(parents=True, exist_ok=True)
    logo         = LeaveOneGroupOut()
    results      = []
    lopo_records = []
    fold_thrs    = []

    for fold, (tr_idx, te_idx) in enumerate(logo.split(hw_X, hw_y, hw_pid)):
        test_pid  = int(hw_pid[te_idx[0]])
        fold_name = names[test_pid] if test_pid < len(names) else f"pid{test_pid}"
        cache_m   = LOPO_DIR / f"fold{fold:03d}_pid{test_pid:03d}.pt"
        cache_t   = LOPO_DIR / f"fold{fold:03d}_pid{test_pid:03d}_thr.pkl"

        X_te_raw, y_te = hw_X[te_idx], hw_y[te_idx]
        X_hw_tr_raw, y_hw_tr = hw_X[tr_idx], hw_y[tr_idx]
        sess_tr = hw_sess[tr_idx]

        # Subsample CHARIS abnormals for this fold
        n_hw_norm = int((y_hw_tr == 0).sum())
        n_hw_abn  = int((y_hw_tr == 1).sum())
        abn_idx   = np.where(charis_y == 1)[0]
        cap       = max(int(MAX_RATIO * n_hw_norm) - n_hw_abn, 50)
        rng_c     = np.random.default_rng(SEED + fold)
        if len(abn_idx) > cap:
            abn_idx = rng_c.choice(abn_idx, size=cap, replace=False)
        X_c_raw = charis_X[abn_idx]
        y_c_raw = charis_y[abn_idx]
        cpid_c  = charis_pid[abn_idx]

        # QT on all train flat features
        X_flat_tr = np.concatenate([X_hw_tr_raw, X_c_raw])
        qt        = fit_qt(X_flat_tr)

        X_hw_tr_qt  = qt.transform(X_hw_tr_raw).astype(np.float32)
        X_c_qt      = qt.transform(X_c_raw).astype(np.float32)
        X_te_qt     = qt.transform(X_te_raw).astype(np.float32)

        # Inner val: last HW train patient (for early stopping)
        train_pids   = np.unique(hw_pid[tr_idx])
        val_pid_inner = train_pids[-1]
        val_mask_inner = hw_pid[tr_idx] == val_pid_inner
        tr_mask_inner  = ~val_mask_inner

        # Build sequences
        hw_pid_tr  = hw_pid[tr_idx]
        hw_sess_tr = hw_sess[tr_idx]

        # Inner train (excluding inner val patient)
        pidx_itr = hw_pid[tr_idx[tr_mask_inner]]
        sess_itr = hw_sess[tr_idx[tr_mask_inner]]
        X_hw_itr_qt = X_hw_tr_qt[tr_mask_inner]
        y_hw_itr    = y_hw_tr[tr_mask_inner]

        # Inner val
        X_hw_iva_qt = X_hw_tr_qt[val_mask_inner]
        y_hw_iva    = y_hw_tr[val_mask_inner]
        sess_iva    = hw_sess_tr[val_mask_inner]
        pidx_iva    = hw_pid_tr[val_mask_inner]

        # Test sequences (within sessions of test patient)
        sess_te = hw_sess[te_idx]
        pidx_te = hw_pid[te_idx]
        X_te_seq, y_te_seq, _ = build_hw_sequences(X_te_qt, y_te, pidx_te, sess_te)

        if len(np.unique(y_te_seq)) < 2 or len(X_te_seq) < SEQ_LEN:
            print(f"  F{fold:3d} {fold_name:<32}  SKIP — test set has single class")
            continue

        if use_cache and cache_m.exists() and cache_t.exists():
            model = ICPClassifier().to(DEVICE)
            model.load_state_dict(torch.load(cache_m, map_location=DEVICE, weights_only=True))
            with open(cache_t, "rb") as f:
                thr = pickle.load(f)
            status = "(c)"
        else:
            # Build train sequences: inner HW train + CHARIS
            X_itr_seq, y_itr_seq, _ = build_hw_sequences(X_hw_itr_qt, y_hw_itr,
                                                          pidx_itr, sess_itr)
            X_c_seq, y_c_seq, _     = build_charis_sequences(X_c_qt, y_c_raw,
                                                              cpid_c, abnormal_only=True)
            X_tr_seq = np.concatenate([X_itr_seq, X_c_seq])
            y_tr_seq = np.concatenate([y_itr_seq, y_c_seq])

            # Build inner val sequences
            X_iva_seq, y_iva_seq, _ = build_hw_sequences(X_hw_iva_qt, y_hw_iva,
                                                          pidx_iva, sess_iva)
            if len(X_iva_seq) < 10 or len(np.unique(y_iva_seq)) < 2:
                # Fall back to 10% of train if inner val patient is unusable
                rng_i = np.random.default_rng(SEED + fold + 1)
                idx_i = rng_i.permutation(len(X_tr_seq)); cut = int(0.9 * len(idx_i))
                X_iva_seq, y_iva_seq = X_tr_seq[idx_i[cut:]], y_tr_seq[idx_i[cut:]]
                X_tr_seq, y_tr_seq   = X_tr_seq[idx_i[:cut]],  y_tr_seq[idx_i[:cut]]

            if len(X_tr_seq) < BATCH:
                print(f"  F{fold:3d} {fold_name:<32}  SKIP — too few train sequences ({len(X_tr_seq)})")
                continue

            model, best_val_auc = train_model(X_tr_seq, y_tr_seq, X_iva_seq, y_iva_seq,
                                               epochs=EPOCHS_LOPO, seed=SEED + fold)
            probs_iva = _predict_probs(model, X_iva_seq)
            thr = youden_threshold(y_iva_seq, probs_iva) if len(np.unique(y_iva_seq)) == 2 else 0.5

            torch.save(model.state_dict(), cache_m)
            with open(cache_t, "wb") as f:
                pickle.dump(thr, f)
            status = f"(t) valAUC={best_val_auc:.3f}"

        probs = _predict_probs(model, X_te_seq)
        preds = (probs >= thr).astype(int)

        auc  = float(roc_auc_score(y_te_seq, probs))
        f1   = float(f1_score(y_te_seq, preds, zero_division=0))
        rec  = float(recall_score(y_te_seq, preds, zero_division=0))
        prec = float(precision_score(y_te_seq, preds, zero_division=0))
        spec = float(recall_score(1 - y_te_seq, 1 - preds, zero_division=0))
        cm   = confusion_matrix(y_te_seq, preds, labels=[0, 1]).tolist()

        print(f"  F{fold:3d} {status}  {fold_name:<32}  "
              f"AUC={auc:.4f}  F1={f1:.3f}  rec={rec:.3f}  spec={spec:.3f}  "
              f"seqs: {int((y_te_seq==0).sum())}neg/{int((y_te_seq==1).sum())}pos")

        results.append({"fold": fold, "patient": fold_name,
                        "auc": round(auc, 4), "f1": round(f1, 4),
                        "recall": round(rec, 4), "precision": round(prec, 4),
                        "specificity": round(spec, 4),
                        "n_test_neg": int((y_te_seq==0).sum()),
                        "n_test_pos": int((y_te_seq==1).sum()),
                        "threshold": round(thr, 4), "confusion_matrix": cm})
        lopo_records.append({"pid": test_pid, "y": y_te_seq.tolist(), "probs": probs.tolist()})
        fold_thrs.append(thr)

    if not results:
        print("  ERROR: no valid LOPO folds."); return [], {}, [], 0.5

    valid_aucs = [r["auc"] for r in results]
    auc_mean   = float(np.mean(valid_aucs))
    auc_std    = float(np.std(valid_aucs))
    ci_lo, ci_hi = float(np.percentile(valid_aucs, 2.5)), float(np.percentile(valid_aucs, 97.5))

    print(f"\n  LOPO AUC  : {auc_mean:.4f} ± {auc_std:.4f}  "
          f"95%CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  LOPO F1   : {float(np.mean([r['f1'] for r in results])):.4f}")
    print(f"  Valid folds: {len(results)}")

    summary = {"auc_mean": round(auc_mean, 4), "auc_std": round(auc_std, 4),
               "auc_ci": [round(ci_lo, 4), round(ci_hi, 4)],
               "f1_mean": round(float(np.mean([r["f1"] for r in results])), 4),
               "n_folds": len(results)}
    return results, summary, lopo_records, float(np.mean(fold_thrs))


# ── Final model ────────────────────────────────────────────────────────────────
def train_final_model(hw_X, hw_y, hw_pid, hw_sess, charis_X, charis_y, charis_pid,
                      mean_thr: float):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    n_hw_norm = int((hw_y == 0).sum()); n_hw_abn = int((hw_y == 1).sum())
    abn_idx   = np.where(charis_y == 1)[0]
    cap       = max(int(MAX_RATIO * n_hw_norm) - n_hw_abn, 50)
    rng       = np.random.default_rng(SEED)
    if len(abn_idx) > cap:
        abn_idx = rng.choice(abn_idx, size=cap, replace=False)
    X_c_raw  = charis_X[abn_idx]; y_c_raw = charis_y[abn_idx]; cpid_c = charis_pid[abn_idx]

    qt = fit_qt(np.concatenate([hw_X, X_c_raw]))
    X_hw_qt = qt.transform(hw_X).astype(np.float32)
    X_c_qt  = qt.transform(X_c_raw).astype(np.float32)

    X_hw_seq, y_hw_seq, _ = build_hw_sequences(X_hw_qt, hw_y, hw_pid, hw_sess)
    X_c_seq,  y_c_seq,  _ = build_charis_sequences(X_c_qt, y_c_raw, cpid_c, abnormal_only=True)
    X_all = np.concatenate([X_hw_seq, X_c_seq])
    y_all = np.concatenate([y_hw_seq, y_c_seq])

    pos, neg = int((y_all==1).sum()), int((y_all==0).sum())
    print(f"\n  Final model: {len(X_all):,} sequences  pos={pos:,} neg={neg:,}  ratio={pos/neg:.2f}:1")

    # Use last 10% as a val set for early stopping
    rng2 = np.random.default_rng(SEED + 999)
    idx  = rng2.permutation(len(X_all)); cut = int(0.9 * len(idx))
    model, val_auc = train_model(X_all[idx[:cut]], y_all[idx[:cut]],
                                  X_all[idx[cut:]], y_all[idx[cut:]],
                                  epochs=EPOCHS_FINAL, seed=SEED)
    print(f"  Best val AUC during training: {val_auc:.4f}")
    print(f"  Threshold (mean LOPO Youden): {mean_thr:.4f}")

    torch.save(model.state_dict(), MODEL_DIR / "bilstm_classifier.pt")
    with open(MODEL_DIR / "bilstm_qt.pkl",  "wb") as f: pickle.dump(qt, f)
    with open(MODEL_DIR / "bilstm_thr.pkl", "wb") as f: pickle.dump(mean_thr, f)
    print(f"  Model -> {MODEL_DIR}/bilstm_classifier.pt")
    return model, qt, mean_thr


# ── Valsalva stats (LOPO predictions) ─────────────────────────────────────────
def run_valsalva_stats(lopo_records: list) -> dict:
    val_means, norm_means = [], []
    for rec in lopo_records:
        y_arr  = np.array(rec["y"]); p_arr = np.array(rec["probs"])
        p1, p0 = p_arr[y_arr==1], p_arr[y_arr==0]
        if len(p1) > 0 and len(p0) > 0:
            val_means.append(float(p1.mean())); norm_means.append(float(p0.mean()))

    if len(val_means) < 4:
        print("  Valsalva: not enough paired subjects (<4)"); return {}

    val_arr = np.array(val_means); norm_arr = np.array(norm_means)
    stat, pval = _wilcoxon(val_arr, norm_arr, alternative="greater")
    pct = 100 * (val_arr > norm_arr).mean()
    sig = "***" if pval<0.001 else ("**" if pval<0.01 else ("*" if pval<0.05 else "ns"))

    SEP = "=" * 65
    print(f"\n{SEP}")
    print("  VALSALVA ANALYSIS (BiLSTM LOPO predictions)")
    print(SEP)
    print(f"  Paired subjects         : {len(val_arr)}")
    print(f"  Valsalva > normal       : {int((val_arr>norm_arr).sum())}/{len(val_arr)}  ({pct:.0f}%)")
    print(f"  Mean P valsalva         : {val_arr.mean():.4f}")
    print(f"  Mean P normal           : {norm_arr.mean():.4f}")
    print(f"  Wilcoxon p              : {pval:.6f}  {sig}")

    return {"n_subjects": len(val_arr), "pct_higher": round(pct,1),
            "mean_valsalva": round(float(val_arr.mean()),4),
            "mean_normal":   round(float(norm_arr.mean()),4),
            "wilcoxon_p":    round(float(pval),6)}


# ── Permutation feature importance ────────────────────────────────────────────
def run_permutation_importance(model: nn.Module, X_seq: np.ndarray,
                                y_seq: np.ndarray) -> dict:
    """
    Baseline AUC → shuffle each feature across all time steps →
    ΔAUC = baseline - shuffled. Larger drop = more important.
    """
    if len(np.unique(y_seq)) < 2: return {}
    baseline_auc = roc_auc_score(y_seq, _predict_probs(model, X_seq))
    results = {}
    rng = np.random.default_rng(SEED)

    SEP = "=" * 65
    print(f"\n{SEP}")
    print("  PERMUTATION FEATURE IMPORTANCE (BiLSTM)")
    print(SEP)

    for fi, feat in enumerate(FEATURES):
        X_perm = X_seq.copy()
        perm   = rng.permutation(len(X_perm))
        X_perm[:, :, fi] = X_perm[perm, :, fi]   # permute across batch, keep time intact
        perm_auc = roc_auc_score(y_seq, _predict_probs(model, X_perm))
        delta    = baseline_auc - perm_auc
        results[feat] = {"baseline_auc": round(float(baseline_auc), 4),
                         "permuted_auc": round(float(perm_auc), 4),
                         "delta":        round(float(delta), 4)}
        bar = "█" * max(0, int(delta * 200))
        print(f"  {feat:<28}: Δ={delta:+.4f}  {bar}")

    return results


# ── Comparison vs XGBoost hybrid ──────────────────────────────────────────────
def compare_vs_xgb(lopo_results: list) -> dict:
    """Wilcoxon on paired per-fold AUCs if hybrid XGBoost results exist."""
    xgb_json = Path("results/hybrid_pipeline/hybrid_results.json")
    if not xgb_json.exists():
        print("  Comparison: hybrid_results.json not found (run hybrid_pipeline_qt.py first)")
        return {}

    with open(xgb_json) as f:
        xgb_data = json.load(f)

    xgb_aucs = [r["auc"] for r in xgb_data.get("lopo_per_fold", [])
                if not np.isnan(r["auc"])]
    bil_aucs = [r["auc"] for r in lopo_results]
    n = min(len(xgb_aucs), len(bil_aucs))
    if n < 4:
        return {}

    a, b = np.array(bil_aucs[:n]), np.array(xgb_aucs[:n])
    try:
        _, p_wx = _wilcoxon(a - b, alternative="two-sided")
    except Exception:
        p_wx = float("nan")

    delta = float(np.mean(a - b))
    winner = "BiLSTM" if delta > 0 else "XGBoost"
    sig = "***" if p_wx<0.001 else ("**" if p_wx<0.01 else ("*" if p_wx<0.05 else "ns"))
    print(f"\n  BiLSTM vs XGBoost Hybrid (Wilcoxon, two-tailed):")
    print(f"  BiLSTM AUC  : {np.mean(a):.4f}  XGBoost AUC : {np.mean(b):.4f}")
    print(f"  ΔAUC        : {delta:+.4f}  Winner: {winner}  p={p_wx:.4f}  {sig}")

    return {"bilstm_auc_mean": round(float(np.mean(a)),4),
            "xgb_auc_mean": round(float(np.mean(b)),4),
            "delta_auc": round(delta,4), "wilcoxon_p": round(float(p_wx),6),
            "winner": winner}


# ── Plots ──────────────────────────────────────────────────────────────────────
def save_plots(results, lopo_records, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-fold AUC
    aucs   = [r["auc"] for r in results]
    labels = [r["patient"].replace(".csv","").replace("icp_","") for r in results]
    fig, ax = plt.subplots(figsize=(max(12, len(aucs)*0.45), 4))
    colors = ["#e74c3c" if a<0.75 else "#f39c12" if a<0.85 else "#2ecc71" for a in aucs]
    ax.bar(range(len(aucs)), aucs, color=colors, alpha=0.85)
    ax.axhline(np.mean(aucs), color="k", ls="--", lw=1.5, label=f"Mean {np.mean(aucs):.4f}")
    ax.set_xticks(range(len(aucs))); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05); ax.set_ylabel("AUC")
    ax.set_title(f"BiLSTM Hybrid LOPO AUC per Patient (seq_len={SEQ_LEN})")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "bilstm_lopo_auc.png", dpi=150, bbox_inches="tight"); plt.close()

    # Probability distribution: valsalva vs normal
    all_val, all_norm = [], []
    for rec in lopo_records:
        y_arr = np.array(rec["y"]); p_arr = np.array(rec["probs"])
        all_val.extend(p_arr[y_arr==1].tolist())
        all_norm.extend(p_arr[y_arr==0].tolist())

    if all_val and all_norm:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(all_norm, bins=40, density=True, alpha=0.6, color="#3498db", label="Normal")
        ax.hist(all_val,  bins=40, density=True, alpha=0.6, color="#e74c3c", label="Valsalva")
        ax.set_xlabel("P(ICP elevated)"); ax.set_ylabel("Density")
        ax.set_title("BiLSTM Score Distribution (LOPO)")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "bilstm_score_dist.png", dpi=150, bbox_inches="tight"); plt.close()

    print(f"  Plots -> {out_dir}/")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    SEP = "=" * 65
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(SEP)
    print(f"  BiLSTM ICP Classifier  |  seq_len={SEQ_LEN}  hidden={HIDDEN}×2  device={DEVICE}")
    print("  Hybrid: CHARIS abn seqs (5:1 cap) + HW sequences")
    print(SEP)

    # [1] Load CHARIS
    for p in [CACHE_X, CACHE_Y, CACHE_PID]:
        if not p.exists():
            print(f"  ERROR: {p} — run full_pipeline_qt.py first."); sys.exit(1)
    charis_X   = np.load(CACHE_X)
    charis_y   = np.load(CACHE_Y)
    charis_pid = np.load(CACHE_PID)
    n_cabn = int((charis_y==1).sum())
    print(f"\n[1] CHARIS: {len(charis_X):,} windows  abnormal={n_cabn:,}  "
          f"patients={len(np.unique(charis_pid))}")

    # [2] Load hardware
    print(f"\n[2] Loading hardware data ...")
    hw_X, hw_y, hw_pid, hw_sess, names = load_hw_labeled(HW_DIR)
    n_pts = len(np.unique(hw_pid))
    n_norm = int((hw_y==0).sum()); n_abn = int((hw_y==1).sum())

    # Quick sequence count preview
    X_prev_qt = fit_qt(hw_X).transform(hw_X).astype(np.float32)
    X_prev, y_prev, _ = build_hw_sequences(X_prev_qt, hw_y, hw_pid, hw_sess)
    n_seq_norm = int((y_prev==0).sum()); n_seq_abn = int((y_prev==1).sum())
    print(f"\n  HW: {n_pts} patients  {n_norm:,} normal windows  {n_abn:,} valsalva windows")
    print(f"  HW sequences: {n_seq_norm:,} normal  {n_seq_abn:,} valsalva  "
          f"(seq_len={SEQ_LEN}, stride={SEQ_STRIDE})")

    # [3] LOPO
    print(f"\n[3] BiLSTM LOPO ({n_pts} folds) ...")
    lopo_results, lopo_m, lopo_records, mean_thr = run_lopo(
        hw_X, hw_y, hw_pid, hw_sess, charis_X, charis_y, charis_pid, names)

    if not lopo_results:
        print("  No valid LOPO folds — check data."); sys.exit(1)

    # [4] Final model
    print(f"\n[4] Training final BiLSTM model ...")
    model, qt, thr = train_final_model(
        hw_X, hw_y, hw_pid, hw_sess, charis_X, charis_y, charis_pid, mean_thr)

    # [5] Valsalva stats
    print(f"\n[5] Valsalva statistics ...")
    val_stats = run_valsalva_stats(lopo_records)

    # [6] Permutation importance (using LOPO predictions pooled)
    print(f"\n[6] Permutation feature importance ...")
    all_probs = np.concatenate([rec["probs"] for rec in lopo_records]).reshape(-1)
    # Need sequences for permutation — build from all HW with fitted final QT
    X_all_qt = qt.transform(hw_X).astype(np.float32)
    X_all_seq, y_all_seq, _ = build_hw_sequences(X_all_qt, hw_y, hw_pid, hw_sess)
    perm_imp = run_permutation_importance(model, X_all_seq, y_all_seq)

    # [7] Compare vs XGBoost hybrid
    print(f"\n[7] Comparing vs XGBoost hybrid ...")
    comp = compare_vs_xgb(lopo_results)

    # [8] Plots
    save_plots(lopo_results, lopo_records, OUT_DIR)

    # Summary
    print(f"\n{SEP}")
    print("  BILSTM CLASSIFIER SUMMARY")
    print(SEP)
    print(f"  HW patients          : {n_pts}")
    print(f"  Seq len / stride     : {SEQ_LEN} / {SEQ_STRIDE}")
    print(f"  LOPO AUC             : {lopo_m['auc_mean']:.4f} ± {lopo_m['auc_std']:.4f}  "
          f"95%CI {lopo_m['auc_ci']}")
    print(f"  LOPO F1              : {lopo_m['f1_mean']:.4f}")
    if val_stats:
        print(f"  Valsalva elevated    : {val_stats['pct_higher']:.0f}%  "
              f"p={val_stats['wilcoxon_p']:.6f}")
    if comp:
        print(f"  vs XGBoost ΔAUC      : {comp['delta_auc']:+.4f}  "
              f"winner={comp['winner']}  p={comp['wilcoxon_p']:.4f}")
    print(SEP)

    # Save JSON
    out = {"date": date.today().isoformat(), "model": "BiLSTM",
           "seq_len": SEQ_LEN, "seq_stride": SEQ_STRIDE,
           "hidden": HIDDEN, "n_layers": N_LAYERS,
           "n_hw_patients": n_pts, "max_charis_ratio": MAX_RATIO,
           "lopo": lopo_m, "lopo_per_fold": lopo_results,
           "valsalva_stats": val_stats,
           "permutation_importance": perm_imp,
           "comparison_vs_xgb": comp}
    out_path = OUT_DIR / "bilstm_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results -> {out_path}")


if __name__ == "__main__":
    main()
