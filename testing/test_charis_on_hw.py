"""TEST A: CHARIS-trained XGBoost (5 features, models/charis_compare/XGBoost) scored on ALL 146 hardware volunteers.
Task on hardware: Valsalva windows = 1, other windows = 0 (there is no ICP ground truth for volunteers).
The model was never trained on hardware data.   Run from repo root:  python testing/test_charis_on_hw.py
"""
import json, pickle, re, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "charis")); import os; os.chdir(ROOT)
import full_pipeline_qt as F
from sklearn.metrics import roc_auc_score
import xgboost  # noqa

d = ROOT / "models/charis_compare/XGBoost"
model = pickle.load(open(d / "model.pkl", "rb")); qt = pickle.load(open(d / "qt_scaler.pkl", "rb")); thr = json.loads((d / "metrics.json").read_text())["threshold"]
rows, allp, ally, allsess, allsub = [], [], [], [], []
for f in sorted(Path("hw-tests").glob("icp_*.csv")):
    m = re.match(r"icp_(\d+)_(\d+)_([MF])", f.name)
    if not m: continue
    X, sess = F.load_hw_csv(f)
    if len(X) == 0 or len(np.unique(sess)) < 2: continue
    p = model.predict_proba(qt.transform(X).astype(np.float32))[:, 1]; y = (sess == 3).astype(int)
    allp.append(p); ally.append(y); allsess.append(sess); allsub.append(np.full(len(p), int(m[1])))
    r = dict(subject=int(m[1]), age=int(m[2]), sex=m[3], windows=len(p))
    if 0 < y.sum() < len(y): r["auc_valsalva_vs_rest"] = roc_auc_score(y, p)
    s0 = sess <= 0
    if y.sum() and (sess == 0).sum(): r["auc_valsalva_vs_supine"] = roc_auc_score(y[(sess == 0) | (sess == 3)], p[(sess == 0) | (sess == 3)])
    r["flag_pct_all"] = 100 * (p >= thr).mean()
    for k, n in {0: "supine", 1: "headup", 2: "headdown", 3: "valsalva"}.items(): r[f"flag_pct_{n}"] = 100 * (p[sess == k] >= thr).mean() if (sess == k).any() else np.nan
    r["valsalva_gt_supine"] = bool(np.nanmean(p[sess == 3]) > np.nanmean(p[sess == 0])) if (sess == 3).any() and (sess == 0).any() else np.nan
    rows.append(r)
df = pd.DataFrame(rows); df.to_csv(OUT / "A_charis_model_on_hw_per_subject.csv", index=False)
p, y, s, sub = map(np.concatenate, (allp, ally, allsess, allsub)); ages = df.set_index("subject").age
pred = p >= thr; tp = (pred & (y == 1)).sum(); fn = (~pred & (y == 1)).sum(); tn = (~pred & (y == 0)).sum(); fp = (pred & (y == 0)).sum()
sup = (s == 0) | (s == 3)
res = dict(n_subjects=len(df), n_windows=int(len(y)), charis_threshold=thr,
           pooled_auc_valsalva_vs_rest=roc_auc_score(y, p), pooled_auc_valsalva_vs_supine=roc_auc_score(y[sup], p[sup]),
           mean_within_subject_auc_vs_rest=float(df.auc_valsalva_vs_rest.mean()), median_within_subject_auc_vs_rest=float(df.auc_valsalva_vs_rest.median()),
           subjects_valsalva_mean_score_above_supine=f"{int(df.valsalva_gt_supine.sum())}/{int(df.valsalva_gt_supine.notna().sum())}",
           sensitivity=float(tp / max(tp + fn, 1)), specificity=float(tn / max(tn + fp, 1)),
           flagged_pct_by_session={n: float(100 * pred[s == k].mean()) for k, n in {0: "supine", 1: "head-up", 2: "head-down", 3: "valsalva"}.items()},
           flagged_pct_age_over_65=float(100 * pred[np.isin(sub, ages[ages > 65].index)].mean()), flagged_pct_age_under_30=float(100 * pred[np.isin(sub, ages[ages < 30].index)].mean()))
o65, u30 = np.isin(sub, ages[ages > 65].index), np.isin(sub, ages[ages < 30].index); k = o65 | u30
res["age_confound_check_auc_over65_vs_under30"] = roc_auc_score(o65[k].astype(int), p[k])
(OUT / "A_charis_model_on_hw.json").write_text(json.dumps(res, indent=1, default=float)); print(json.dumps(res, indent=1, default=float))
