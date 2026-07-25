"""
anomaly_validation.py
=====================
Internal-consistency (NOT ground-truth) check for the 146-subject hardware
ICP-proxy dataset.

Question
--------
Do unsupervised anomaly detectors, trained ONLY on the 139 subjects the CHARIS
model did NOT flag, independently single out the 7 CHARIS-flagged subjects — and
does any such agreement SURVIVE controlling for age?

The 7 CHARIS flags are a WEAK REFERENCE PRIOR, never ground truth. Variable
names use `charis_prior_flag`. No invasive ICP ground truth exists.

Methods (each trained on the 139 prior-normal, scored on all 146):
  a. Isolation Forest
  b. One-Class SVM (RBF)
  c. XGBoost synthetic-outlier (column-shuffled negatives break joint structure)

Then: age-confound control (Spearman age~score; age-residualised re-run;
within-age-band subgroups) and a within-subject dose-response cross-check that
needs no labels at all.

Output: results/anomaly_validation/  (report.md, *.png, metrics.json)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import xgboost as xgb

import hybrid_pipeline_v4 as H

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path("results/anomaly_validation"); OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "hw_subject_cache.npz"
SEED = 42
FEATURES = H.FEATURES
ORDER = [1, 0, 2, 3]                     # ascending expected ICP
SESSION_NAME = {1: "head-up", 0: "supine", 2: "head-down", 3: "valsalva"}
SLOW_IDX = FEATURES.index("slow_wave_power")


# ── 1. Load per-subject aggregate + per-window matrix ────────────────────────
def load_subjects():
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return (z["S"], z["ages"], z["flag"], list(z["names"]),
                z["slow_by_sess"], z["Xw"], z["pidw"], z["sessw"])
    flags = json.load(open(H.FLAGS_PATH)); flags = flags.get("patients", flags)
    Xw, yw, pidw, sessw, names = H.load_hw_labeled(H.HW_DIR, flags)
    n = len(names)
    S = np.zeros((n, len(FEATURES)), np.float64)          # per-subject mean feature vec
    slow_by_sess = np.full((n, 4), np.nan)                # subject x [head-up,supine,head-down,valsalva]
    ages = np.zeros(n); flag = np.zeros(n, int)
    for i, nm in enumerate(names):
        m = pidw == i
        S[i] = Xw[m].mean(axis=0)
        ages[i] = int(nm.split("_")[2])
        flag[i] = 1 if flags[nm]["flagged"] else 0
        for c, s in enumerate(ORDER):
            sm = m & (sessw == s)
            if sm.sum() > 0:
                slow_by_sess[i, c] = Xw[sm, SLOW_IDX].mean()
    np.savez(CACHE, S=S, ages=ages, flag=flag, names=np.array(names, object),
             slow_by_sess=slow_by_sess, Xw=Xw, pidw=pidw, sessw=sessw)
    return S, ages, flag, names, slow_by_sess, Xw, pidw, sessw


# ── standardise using ONLY the prior-normal subjects (no leakage from the 7) ──
def zscore(fit_rows, apply_rows):
    mu, sd = fit_rows.mean(0), fit_rows.std(0) + 1e-9
    return (apply_rows - mu) / sd


# ── 2. Three anomaly detectors: fit on normal rows, score all rows ───────────
def iforest_scores(Xn, Xall):
    m = IsolationForest(n_estimators=400, contamination="auto", random_state=SEED)
    m.fit(Xn)
    return -m.score_samples(Xall)          # higher = more anomalous

def ocsvm_scores(Xn, Xall):
    m = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
    m.fit(Xn)
    return -m.decision_function(Xall)       # higher = more anomalous

def xgb_synthetic_scores(Xn, Xall, n_rep=8):
    rng = np.random.default_rng(SEED)
    reps = []
    for _ in range(n_rep):
        sh = Xn.copy()
        for j in range(sh.shape[1]):
            sh[:, j] = sh[rng.permutation(len(sh)), j]   # break joint structure
        reps.append(sh)
    Xtr = np.vstack([Xn] + reps)
    ytr = np.concatenate([np.ones(len(Xn)), np.zeros(sum(len(r) for r in reps))])
    d = xgb.DMatrix(Xtr, label=ytr, feature_names=FEATURES)
    bst = xgb.train({"objective": "binary:logistic", "eta": 0.1, "max_depth": 3,
                     "subsample": 0.8, "colsample_bytree": 0.8, "lambda": 2.0,
                     "seed": SEED, "verbosity": 0},
                    d, num_boost_round=200)
    p_real = bst.predict(xgb.DMatrix(Xall, feature_names=FEATURES))
    return 1.0 - p_real                     # low P(real) -> anomalous

METHODS = {"IsolationForest": iforest_scores,
           "OneClassSVM": ocsvm_scores,
           "XGB-synthetic": xgb_synthetic_scores}


def run_methods(S, flag, standardise=True):
    """Fit each detector on prior-normal rows, score all rows."""
    Xn = S[flag == 0]
    if standardise:
        mu, sd = Xn.mean(0), Xn.std(0) + 1e-9
        Xn_s, Xall_s = (Xn - mu) / sd, (S - mu) / sd
    else:
        Xn_s, Xall_s = Xn, S
    return {name: fn(Xn_s, Xall_s) for name, fn in METHODS.items()}


# ── metrics helpers ──────────────────────────────────────────────────────────
def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 0.0

def agreement(scores, flag, names, k=7):
    from sklearn.metrics import roc_auc_score, average_precision_score
    order = np.argsort(-scores)
    topk = set(order[:k].tolist())
    flagged = set(np.where(flag == 1)[0].tolist())
    return {
        "auc_vs_prior": round(float(roc_auc_score(flag, scores)), 3),
        "ap_vs_prior": round(float(average_precision_score(flag, scores)), 3),
        "top7_names": [names[i] for i in order[:7]],
        "top10_names": [names[i] for i in order[:10]],
        "jaccard_top7": round(jaccard(topk, flagged), 3),
        "jaccard_top10": round(jaccard(set(order[:10].tolist()), flagged), 3),
        "n_flag_in_top7": len(topk & flagged),
    }


# ── 4. Age residualisation: remove normal-aging trend, re-run ────────────────
def residualise(S, ages, flag):
    """Fit feature~age linear trend on prior-NORMAL subjects, subtract from all."""
    R = np.zeros_like(S)
    an = ages[flag == 0]
    for j in range(S.shape[1]):
        b, a = np.polyfit(an, S[flag == 0, j], 1)     # slope, intercept on normals
        R[:, j] = S[:, j] - (a + b * ages)
    return R


# ── 5. Dose-response cross-check (label-free, slow_wave_power) ────────────────
def dose_response(slow_by_sess):
    """Within-subject Spearman(ICP rank, slow_wave_power) per subject.
    ORDER cols already ascending ICP: head-up<supine<head-down<valsalva."""
    rhos = np.full(len(slow_by_sess), np.nan)
    for i, vec in enumerate(slow_by_sess):
        if np.all(np.isfinite(vec)):
            rhos[i] = spearmanr([0, 1, 2, 3], vec)[0]
    return rhos


# ── plots ────────────────────────────────────────────────────────────────────
def plot_all(S, ages, flag, names, scores, scores_res, rhos, slow_by_sess):
    # score distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (nm, sc) in zip(axes, scores.items()):
        ax.hist(sc[flag == 0], bins=25, alpha=0.6, label="prior-normal (139)", color="#3498db")
        for x in sc[flag == 1]:
            ax.axvline(x, color="#e74c3c", lw=1.2, alpha=0.8)
        ax.set_title(f"{nm}\n(red lines = 7 CHARIS-prior subjects)", fontsize=9)
        ax.set_xlabel("anomaly score"); ax.legend(fontsize=7)
    plt.suptitle("Outlier-score distributions — flagged subjects vs cohort")
    plt.tight_layout(); plt.savefig(OUT / "score_distributions.png", dpi=140); plt.close()

    # age vs score
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (nm, sc) in zip(axes, scores.items()):
        ax.scatter(ages[flag == 0], sc[flag == 0], s=18, c="#3498db", label="prior-normal")
        ax.scatter(ages[flag == 1], sc[flag == 1], s=55, c="#e74c3c", marker="D", label="CHARIS-prior")
        rho = spearmanr(ages, sc)[0]
        ax.set_title(f"{nm}\nSpearman(age, score)={rho:+.2f}", fontsize=9)
        ax.set_xlabel("age"); ax.set_ylabel("anomaly score"); ax.legend(fontsize=7)
    plt.suptitle("Anomaly score vs AGE — is the detector just finding old subjects?")
    plt.tight_layout(); plt.savefig(OUT / "age_vs_score.png", dpi=140); plt.close()

    # residualised age vs score
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (nm, sc) in zip(axes, scores_res.items()):
        ax.scatter(ages[flag == 0], sc[flag == 0], s=18, c="#3498db", label="prior-normal")
        ax.scatter(ages[flag == 1], sc[flag == 1], s=55, c="#e74c3c", marker="D", label="CHARIS-prior")
        rho = spearmanr(ages, sc)[0]
        ax.set_title(f"{nm} (age-residualised)\nSpearman(age, score)={rho:+.2f}", fontsize=9)
        ax.set_xlabel("age"); ax.set_ylabel("anomaly score"); ax.legend(fontsize=7)
    plt.suptitle("After removing the normal-aging feature trend")
    plt.tight_layout(); plt.savefig(OUT / "age_vs_score_residualised.png", dpi=140); plt.close()

    # dose-response flagged vs non-flagged
    fig, ax = plt.subplots(figsize=(7, 5))
    ok = np.isfinite(rhos)
    ax.hist(rhos[ok & (flag == 0)], bins=15, alpha=0.6, color="#3498db",
            label=f"non-flagged (n={int((ok&(flag==0)).sum())})", density=True)
    for x in rhos[ok & (flag == 1)]:
        ax.axvline(x, color="#e74c3c", lw=1.4, alpha=0.85)
    ax.set_xlabel("within-subject Spearman(ICP ladder, slow_wave_power)")
    ax.set_ylabel("density")
    ax.set_title("Dose-response consistency (red lines = 7 CHARIS-prior subjects)")
    ax.legend(); plt.tight_layout(); plt.savefig(OUT / "dose_response.png", dpi=140); plt.close()


def main():
    S, ages, flag, names, slow_by_sess, Xw, pidw, sessw = load_subjects()
    n = len(names)
    print(f"Loaded {n} subjects  |  CHARIS-prior flagged = {int(flag.sum())}")
    print(f"Ages: flagged {sorted(ages[flag==1].astype(int))} | "
          f"cohort median {np.median(ages):.0f} (range {ages.min():.0f}-{ages.max():.0f})")

    # 2-3. run methods + agreement
    scores = run_methods(S, flag)
    agree = {nm: agreement(sc, flag, names) for nm, sc in scores.items()}

    # method-method overlap (top-7 sets)
    top7 = {nm: set(np.argsort(-sc)[:7].tolist()) for nm, sc in scores.items()}
    pair_jac = {}
    ms = list(scores)
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            pair_jac[f"{ms[i]} vs {ms[j]}"] = round(jaccard(top7[ms[i]], top7[ms[j]]), 3)

    # 4. age confound
    age_rho = {nm: round(float(spearmanr(ages, sc)[0]), 3) for nm, sc in scores.items()}
    R = residualise(S, ages, flag)
    scores_res = run_methods(R, flag)
    agree_res = {nm: agreement(sc, flag, names) for nm, sc in scores_res.items()}
    age_rho_res = {nm: round(float(spearmanr(ages, sc)[0]), 3) for nm, sc in scores_res.items()}

    # subgroup: within elderly (>=60) and young (<60)
    def subgroup(mask, label):
        idx = np.where(mask)[0]
        sub_flag = flag[idx]
        out = {"label": label, "n": int(mask.sum()), "n_flag": int(sub_flag.sum())}
        if sub_flag.sum() >= 1 and (sub_flag == 0).sum() >= 5:
            sc = IsolationForest(n_estimators=400, contamination="auto",
                                 random_state=SEED)
            Xn = zscore(S[idx][sub_flag == 0], S[idx][sub_flag == 0])
            sc.fit(Xn)
            allz = zscore(S[idx][sub_flag == 0], S[idx])
            an = -sc.score_samples(allz)
            order = np.argsort(-an)
            flagpos = set(np.where(sub_flag == 1)[0].tolist())
            out["n_flag_in_top_nflag"] = len(set(order[:sub_flag.sum()].tolist()) & flagpos)
            out["auc_vs_prior"] = round(float(
                __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(sub_flag, an)), 3) \
                if len(np.unique(sub_flag)) == 2 else None
        return out
    elderly = subgroup(ages >= 60, "elderly>=60")
    young = subgroup(ages < 60, "young<60")

    # 5. dose-response
    rhos = dose_response(slow_by_sess)
    ok = np.isfinite(rhos)
    dr_flag = rhos[ok & (flag == 1)]
    dr_norm = rhos[ok & (flag == 0)]
    mw = mannwhitneyu(dr_norm, dr_flag, alternative="two-sided") if len(dr_flag) >= 1 else None
    dose = {
        "metric": "within-subject Spearman(ICP ladder, slow_wave_power)",
        "n_flagged_with_all4": int(len(dr_flag)),
        "n_nonflag_with_all4": int(len(dr_norm)),
        "mean_rho_flagged": round(float(np.mean(dr_flag)), 3) if len(dr_flag) else None,
        "mean_rho_nonflagged": round(float(np.mean(dr_norm)), 3) if len(dr_norm) else None,
        "mannwhitney_p": round(float(mw.pvalue), 4) if mw else None,
    }

    plot_all(S, ages, flag, names, scores, scores_res, rhos, slow_by_sess)

    charis7 = [names[i] for i in np.where(flag == 1)[0]]
    out = {
        "n_subjects": n, "n_charis_prior_flag": int(flag.sum()),
        "charis_prior_flag_subjects": charis7,
        "ages_flagged": sorted(ages[flag == 1].astype(int).tolist()),
        "agreement_raw": agree,
        "method_overlap_top7_jaccard": pair_jac,
        "age_spearman_raw": age_rho,
        "agreement_age_residualised": agree_res,
        "age_spearman_residualised": age_rho_res,
        "subgroups": {"elderly": elderly, "young": young},
        "dose_response": dose,
    }
    json.dump(out, open(OUT / "metrics.json", "w"), indent=2)

    # ── console summary ──
    print("\n=== AGREEMENT WITH CHARIS PRIOR (weak reference, NOT accuracy) ===")
    for nm in scores:
        a = agree[nm]
        print(f"  {nm:<16} AUC={a['auc_vs_prior']:.3f} AP={a['ap_vs_prior']:.3f} "
              f"| {a['n_flag_in_top7']}/7 in top-7  Jaccard={a['jaccard_top7']:.3f}")
    print("\n=== METHOD-METHOD OVERLAP (top-7 Jaccard) ===")
    for k, v in pair_jac.items():
        print(f"  {k:<34} {v:.3f}")
    print("\n=== AGE CONFOUND ===")
    for nm in scores:
        print(f"  {nm:<16} Spearman(age,score) raw={age_rho[nm]:+.2f} -> "
              f"residualised={age_rho_res[nm]:+.2f} | "
              f"AUC-vs-prior {agree[nm]['auc_vs_prior']:.3f} -> {agree_res[nm]['auc_vs_prior']:.3f} "
              f"| {agree_res[nm]['n_flag_in_top7']}/7 survive")
    print("\n=== SUBGROUP (age-homogeneous) ===")
    for g in (elderly, young):
        print(f"  {g['label']:<14} n={g['n']} flagged={g['n_flag']} "
              f"AUC={g.get('auc_vs_prior')} top-hit={g.get('n_flag_in_top_nflag')}")
    print("\n=== DOSE-RESPONSE CROSS-CHECK (label-free) ===")
    print(f"  mean within-subject rho:  flagged={dose['mean_rho_flagged']}  "
          f"non-flagged={dose['mean_rho_nonflagged']}  MW-p={dose['mannwhitney_p']}")
    print(f"\nSaved -> {OUT}/  (metrics.json + 4 PNGs)")
    return out


if __name__ == "__main__":
    main()
