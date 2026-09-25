"""Hunt for leakage that a subject-grouped split would NOT catch (hardware, Valsalva vs rest, XGBoost).

    python hardware-vasalva-method/leak_hunt.py

  1. duplicate recordings : do any two files share identical raw sample runs (same recording saved twice under different IDs)?
  2. recording-batch holdout: hold out whole blocks of consecutive subject IDs (recorded around the same time/device state), train on the rest.
  3. early -> late / late -> early: train on the first half of subject IDs, test on the second half and the reverse.
If the 0.99 came from batch/device/time leakage these would fall; a genuine subject-independent effect would hold.
"""
from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import common as C

X, y, P, S, age, names = C.load(); num = np.array([int(re.match(r"icp_(\d+)_", n).group(1)) for n in names]); out = {}
# ---- 1. duplicate recordings (content-defined sampling of 8-sample runs of ir_raw, robust to time offsets)
rng = np.random.default_rng(1); w = rng.integers(1, 2**62, 8, dtype=np.int64); table = defaultdict(set)
for i, n in enumerate(names):
    v = pd.read_csv(Path("hw-tests") / n, comment="#", usecols=["ir_raw"], low_memory=False).ir_raw.to_numpy(np.int64)
    win = np.lib.stride_tricks.sliding_window_view(v, 8); ok = win.std(1) > 0; h = (win * w).sum(1); pick = ok & (h % 64 == 0)
    for hv in np.unique(h[pick]): table[int(hv)].add(i)
pairs = defaultdict(int)
for hv, fs in table.items():
    if 1 < len(fs) <= 6:
        fs = sorted(fs)
        for a in range(len(fs)):
            for b in range(a + 1, len(fs)): pairs[(fs[a], fs[b])] += 1
top = sorted(pairs.items(), key=lambda kv: -kv[1])[:5]
out["duplicate_check"] = {"n_files": len(names), "largest_shared_run_counts": [(names[a], names[b], int(c)) for (a, b), c in top], "pairs_with_>=20_shared": int(sum(c >= 20 for c in pairs.values()))}
print("duplicates: pairs with >=20 shared runs:", out["duplicate_check"]["pairs_with_>=20_shared"], "| top:", out["duplicate_check"]["largest_shared_run_counts"][:3], flush=True)


def heldout_auc(test_subj, seed):
    tr_pool = np.setdiff1d(np.unique(P), test_subj); va_s = np.random.default_rng(seed).choice(tr_pool, C.N_VAL, replace=False)
    va = np.where(np.isin(P, va_s))[0]; fit = np.where(np.isin(P, np.setdiff1d(tr_pool, va_s)))[0]; te = np.where(np.isin(P, test_subj))[0]
    assert not set(P[fit]) & set(P[te]); m, qt, thr, it = C.fit_scaled("XGBoost", X, y, fit, va); s = C.prob(m, qt, X[te]); mm = C.metrics(y[te], s, thr)
    return dict(n_test_subjects=int(len(test_subj)), auc=mm["auc"], sensitivity=mm["sensitivity"], specificity=mm["specificity"])


order = np.unique(P)[np.argsort([num[p] for p in np.unique(P)])]; blocks = np.array_split(order, 5); out["id_block_holdout"] = []
for b, blk in enumerate(blocks):
    r = heldout_auc(blk, b); r["subject_numbers"] = f"{int(min(num[q] for q in blk))}-{int(max(num[q] for q in blk))}"; out["id_block_holdout"].append(r); print("block", r, flush=True)
h = len(order) // 2; out["early_to_late"] = heldout_auc(order[h:], 11); out["late_to_early"] = heldout_auc(order[:h], 12); print("early->late", out["early_to_late"], "\nlate->early", out["late_to_early"], flush=True)
(C.OUT / "leak_hunt.json").write_text(json.dumps(out, indent=1))
