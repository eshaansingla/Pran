import glob
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import QuantileTransformer
from coral_lopo import coral_align, youden_threshold

FEATURES = ["cardiac_amplitude", "cardiac_frequency", "respiratory_amplitude", "slow_wave_power", "cardiac_power"]
SESS = {0: "supine", 1: "head-up", 2: "head-down", 3: "valsalva"}

Xc = np.load("results/audit/cache/X.npy").astype(np.float64)
yc = np.load("results/audit/cache/y.npy").astype(int)
Xh_bg = np.load("support/results/hw_features_cache.npz")["X"].astype(np.float64)

qt_c = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=0)
qt_h = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=0)
Xc_qt = qt_c.fit_transform(Xc)
Xh_qt = qt_h.fit_transform(Xh_bg)
cov_target = np.cov(Xh_qt.T)
Xc_al = coral_align(Xc_qt, cov_target)

n_pos, n_neg = yc.sum(), len(yc) - yc.sum()
clf = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, scale_pos_weight=n_neg / n_pos, eval_metric="auc",
                         random_state=0, n_jobs=-1)
clf.fit(Xc_al, yc)
p_all = clf.predict_proba(Xc_al)[:, 1]
thr = youden_threshold(yc, p_all)
print(f"threshold={thr:.3f}\n")

rows = []
per_subject = []
for f in sorted(glob.glob("hw_data/*_features.csv")):
    name = os.path.basename(f)
    df = pd.read_csv(f)
    X_ = df[FEATURES].to_numpy(np.float64)
    Xq = qt_h.transform(X_)
    p = clf.predict_proba(Xq.astype(np.float32))[:, 1]
    df["p"] = p
    df["flag"] = (p >= thr).astype(int)
    for sess, g in df.groupby("session_label"):
        rows.append(dict(subject=name, session=SESS.get(sess, f"label{sess}"), n=len(g),
                          flagged_pct=100 * g["flag"].mean(), mean_p=g["p"].mean()))
    per_subject.append(dict(subject=name, n=len(df), flagged_pct=100 * df["flag"].mean(), mean_p=df["p"].mean()))

res = pd.DataFrame(rows)
res.to_csv("charis/hw_coral_session_breakdown.csv", index=False)
subj = pd.DataFrame(per_subject)
subj.to_csv("charis/hw_coral_per_subject.csv", index=False)

print("=== per-session summary (across all 146 subjects) ===")
print(res.groupby("session").agg(subjects=("subject", "nunique"), n_windows=("n", "sum"),
                                   mean_flagged_pct=("flagged_pct", "mean"),
                                   median_flagged_pct=("flagged_pct", "median"),
                                   mean_p=("mean_p", "mean")).round(2))

# within-subject: does valsalva score higher than that same subject's normal/supine?
piv = res.pivot_table(index="subject", columns="session", values="flagged_pct")
if "valsalva" in piv.columns and "supine" in piv.columns:
    both = piv.dropna(subset=["valsalva", "supine"])
    up = (both["valsalva"] > both["supine"]).sum()
    print(f"\nWithin-subject: valsalva flagged% > supine flagged% in {up}/{len(both)} subjects "
          f"({100*up/len(both):.1f}%)")
    print(f"mean within-subject delta (valsalva - supine): {(both['valsalva']-both['supine']).mean():+.2f} pct pts")

print(f"\n=== overall per-subject flagged% distribution (n={len(subj)}) ===")
print(subj["flagged_pct"].describe().round(2))
