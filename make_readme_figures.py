"""
make_readme_figures.py
======================
Regenerate every figure the README embeds, at the 146-subject cohort, with a
single consistent colourblind-safe (Okabe-Ito) style. Reads only from saved
artifacts (no retraining):

  results/hybrid_pipeline_v4/lopo_records.pkl   hybrid per-window LOPO probs
  results/hybrid_pipeline_v4/results_v4.json    hybrid metrics
  results/hw_charis_records.pkl                 pure-CHARIS zero-shot probs on HW
  results/qt_pipeline/qt_results.json           CHARIS ground-truth metrics
  results/two_model_comparison.json             merged headline

Outputs -> assets/  (tracked in git for the README)
"""
from __future__ import annotations
import json, pickle, re, glob, os
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150})

# Okabe-Ito
BLUE, ORANGE, GREEN, RED = "#0072B2", "#E69F00", "#009E73", "#D55E00"
SKY, YELLOW, PURPLE, GREY = "#56B4E9", "#F0E442", "#CC79A7", "#999999"

ASSETS = Path("assets"); ASSETS.mkdir(exist_ok=True)
ORDERED = [1, 0, 2, 3]
SNAME = {1: "head-up\n30°", 0: "supine", 2: "head-down\n10°", 3: "Valsalva"}


def L(p):
    return pickle.load(open(p, "rb"))


def J(p):
    return json.load(open(p))


hyb = L("results/hybrid_pipeline_v4/lopo_records.pkl")
cha = L("results/hw_charis_records.pkl")
res = J("results/hybrid_pipeline_v4/results_v4.json")
qt  = J("results/qt_pipeline/qt_results.json")


def ladder(records):
    rows = []
    for r in records:
        s = np.array(r["sessions"]); p = np.array(r["probs"])
        per = {k: float(p[s == k].mean()) for k in ORDERED if (s == k).sum() > 0}
        if len(per) == 4:
            rows.append([per[k] for k in ORDERED])
    return np.array(rows)


# ── FIG 1 · Headline two-model dose-response ladder ───────────────────────────
def fig_two_model():
    Ma, Mb = ladder(cha), ladder(hyb)
    cmp = J("results/two_model_comparison.json")
    rho_a = cmp["model_A_pure_charis"]["hardware_dose_response"]["mean_within_subject_spearman"]
    rho_b = cmp["model_B_hybrid_v4"]["hardware_dose_response"]["mean_within_subject_spearman"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, M, tag, col, rho in [
        (axes[0], Ma, "Model A · pure-CHARIS  (ZERO-SHOT, never saw hardware)", BLUE, rho_a),
        (axes[1], Mb, "Model B · hybrid V4  (LOPO)", ORANGE, rho_b)]:
        base = M[:, 1:2]
        Mn = M - base
        for row in Mn:
            ax.plot(range(4), row, color=GREY, alpha=0.15, lw=0.8)
        mean = Mn.mean(0)
        ax.plot(range(4), mean, color=col, lw=3.5, marker="o", ms=10,
                label="cohort mean", zorder=5)
        ax.axhline(0, color="k", lw=0.7, ls=":")
        ax.set_xticks(range(4)); ax.set_xticklabels([SNAME[k] for k in ORDERED])
        ax.set_title(f"{tag}\nmean within-subject Spearman ρ = {rho:+.2f}   (n={M.shape[0]})",
                     fontsize=10)
        ax.set_ylabel("Model output − supine baseline")
    fig.suptitle("Within-subject ICP dose–response ladder — two independent models converge\n"
                 "(head-up < supine < head-down < Valsalva)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(ASSETS / "two_model_dose_response.png", bbox_inches="tight")
    plt.close()


# ── FIG 2 · Hybrid dose-response spaghetti with significance ──────────────────
def fig_dose_response():
    M = ladder(hyb)
    Mn = M - M[:, 1:2]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for row in Mn:
        ax.plot(range(4), row, color=GREY, alpha=0.16, lw=0.8)
    mean, se = Mn.mean(0), Mn.std(0) / np.sqrt(len(Mn))
    ax.errorbar(range(4), mean, yerr=se, color=ORANGE, lw=3.5, marker="o",
                ms=11, capsize=5, label="cohort mean ± SE", zorder=5)
    ax.axhline(0, color="k", lw=0.7, ls=":")
    dr = res["dose_response"]
    pairs = dr["adjacent_pairs"]
    def star(p): return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
    ymax = mean.max()
    for i, pr in enumerate(pairs):
        ax.text(i + 0.5, ymax * 1.05 + 0.005, star(pr["wilcoxon_p"]),
                ha="center", fontsize=13, color=RED if pr["wilcoxon_p"] < .05 else GREY)
    ax.set_xticks(range(4)); ax.set_xticklabels([SNAME[k] for k in ORDERED])
    ax.set_ylabel("Model output − supine baseline")
    ax.set_title(f"Hybrid V4 — within-subject ICP dose–response  (N={len(M)})\n"
                 f"Friedman χ²={dr['friedman_chi2']:.0f}, p<10⁻¹⁶ · "
                 f"3-level ρ={dr['three_level']['mean_within_subject_spearman']:+.2f}",
                 fontsize=11)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(ASSETS / "dose_response.png", bbox_inches="tight")
    plt.close()


# ── FIG 3 · Pooled LOPO ROC (both models where honest) ────────────────────────
def fig_roc():
    ay = np.array([v for r in hyb for v in r["y"]])
    ap = np.array([v for r in hyb for v in r["probs"]])
    fpr, tpr, _ = roc_curve(ay, ap)
    auc = res["lopo_eval"]["pooled_auc"]
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.plot(fpr, tpr, color=ORANGE, lw=3,
            label=f"Hybrid V4 pooled LOPO (AUC={auc:.3f})")
    # CHARIS ground-truth LOPO mean AUC as a reference marker line
    ax.plot([0, 1], [0, 1], color=GREY, lw=1.2, ls="--", label="chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Pooled Leave-One-Patient-Out ROC — 146 hardware subjects\n"
                 "(labels are pure-CHARIS pseudo-labels → consistency metric)",
                 fontsize=10)
    ax.legend(loc="lower right"); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(ASSETS / "lopo_roc.png", bbox_inches="tight")
    plt.close()


# ── FIG 4 · Feature importance + ablation ─────────────────────────────────────
def fig_features():
    fi = res["feature_importance"]; ab = res["feature_ablation"]
    order = sorted(fi, key=lambda k: fi[k])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    axes[0].barh(range(len(order)), [fi[k] for k in order], color=BLUE, alpha=0.85)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([k.replace("_", " ") for k in order])
    for i, k in enumerate(order):
        axes[0].text(fi[k] + 0.008, i, f"{fi[k]*100:.1f}%", va="center", fontsize=9)
    axes[0].set_xlabel("Normalised gain")
    axes[0].set_title("(a) Feature importance — hybrid V4", fontsize=11)
    axes[0].set_xlim(0, max(fi.values()) * 1.18)

    full = res["lopo_eval"]["pooled_auc"]
    aborder = sorted(ab, key=lambda k: ab[k]["delta"])  # most harmful first
    deltas = [-ab[k]["delta"] for k in aborder]  # AUC drop when removed (positive = important)
    axes[1].barh(range(len(aborder)), deltas, color=RED, alpha=0.85)
    axes[1].set_yticks(range(len(aborder)))
    axes[1].set_yticklabels([k.replace("_", " ") for k in aborder])
    for i, k in enumerate(aborder):
        axes[1].text(-ab[k]["delta"] + 0.002, i, f"−{-ab[k]['delta']:.3f}",
                     va="center", fontsize=9)
    axes[1].set_xlabel("AUC lost when feature removed")
    axes[1].set_title(f"(b) Drop-one ablation  (full LOPO AUC={full:.3f})", fontsize=11)
    axes[1].set_xlim(0, max(deltas) * 1.25)
    plt.tight_layout()
    plt.savefig(ASSETS / "feature_analysis.png", bbox_inches="tight")
    plt.close()


# ── FIG 5 · Per-patient LOPO scores ───────────────────────────────────────────
def fig_patient_scores():
    recs = sorted(hyb, key=lambda r: r["mean_prob"])
    scores = [r["mean_prob"] for r in recs]
    cols = [RED if r["true_label"] == 1 else BLUE for r in recs]
    thr = 0.2953
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(range(len(recs)), scores, color=cols, alpha=0.9)
    ax.axhline(thr, color="k", ls="--", lw=1, label=f"CHARIS threshold ({thr})")
    ax.set_xlabel("Hardware subject (sorted by score)")
    ax.set_ylabel("Mean P(ICP elevated)")
    ax.set_title("Per-subject hybrid LOPO score — 146 subjects "
                 "(7 flagged abnormal, all elderly 70–83)")
    ax.legend(handles=[Patch(color=RED, label="Flagged abnormal (n=7)"),
                       Patch(color=BLUE, label="Normal (n=139)"),
                       plt.Line2D([0], [0], color="k", ls="--", label=f"threshold {thr}")],
              loc="upper left")
    ax.set_xlim(-1, len(recs))
    plt.tight_layout()
    plt.savefig(ASSETS / "patient_scores.png", bbox_inches="tight")
    plt.close()


# ── FIG 6 · CHARIS model comparison (ground truth) ────────────────────────────
def fig_charis_comparison():
    lo = qt["lopo"]; bl = qt["baselines_lopo"]
    models = ["XGBoost", "RandForest", "LogReg", "LinearSVM"]
    aucs = [lo["auc_mean"], bl["RandForest"]["auc_mean"],
            bl["LogReg"]["auc_mean"], bl["LinearSVM"]["auc_mean"]]
    errs = [lo["auc_std"], bl["RandForest"]["auc_std"],
            bl["LogReg"]["auc_std"], bl["LinearSVM"]["auc_std"]]
    cols = [GREEN, BLUE, ORANGE, PURPLE]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    b = ax.bar(models, aucs, yerr=errs, color=cols, alpha=0.9, capsize=5)
    for rect, a in zip(b, aucs):
        ax.text(rect.get_x() + rect.get_width()/2, a + 0.01, f"{a:.3f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0.5, color=GREY, ls="--", lw=1)
    ax.set_ylim(0.5, 1.02); ax.set_ylabel("LOPO AUC (13 CHARIS patients)")
    ax.set_title("Clinical foundation model — invasive-ICP ground truth\n"
                 "XGBoost vs baselines (Leave-One-Patient-Out)", fontsize=11)
    plt.tight_layout()
    plt.savefig(ASSETS / "charis_model_comparison.png", bbox_inches="tight")
    plt.close()


# ── FIG 7 · Cohort demographics ───────────────────────────────────────────────
def fig_demographics():
    ages, sexes = [], []
    for f in sorted(glob.glob("hw-tests/*.csv")):
        m = re.match(r"icp_(\d+)_(\d+)_([MF])", os.path.basename(f))
        if m:
            ages.append(int(m.group(2))); sexes.append(m.group(3))
    ages = np.array(ages)
    groups = [("Children\n<13", 0, 12), ("Teens\n13-18", 13, 18),
              ("Young adult\n19-25", 19, 25), ("Adult\n26-45", 26, 45),
              ("Middle\n46-64", 46, 64), ("Elderly\n65+", 65, 200)]
    counts = [int(((ages >= lo) & (ages <= hi)).sum()) for _, lo, hi in groups]
    abn_ct = [0, 0, 0, 0, 0, 7]  # all 7 abnormal are elderly
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    x = range(len(groups))
    axes[0].bar(x, counts, color=BLUE, alpha=0.9, label="normal")
    axes[0].bar(x, abn_ct, color=RED, alpha=0.95, label="flagged abnormal")
    axes[0].set_xticks(x); axes[0].set_xticklabels([g[0] for g in groups])
    for i, c in enumerate(counts):
        axes[0].text(i, c + 0.7, str(c), ha="center", fontsize=10)
    axes[0].set_ylabel("Subjects"); axes[0].legend()
    axes[0].set_title(f"(a) Age distribution  (N={len(ages)}, median "
                      f"{int(np.median(ages))}, range {ages.min()}–{ages.max()})", fontsize=10)
    m, f = sexes.count("M"), sexes.count("F")
    axes[1].bar(["Male", "Female"], [m, f], color=[SKY, ORANGE], alpha=0.9)
    for i, c in enumerate([m, f]):
        axes[1].text(i, c + 1, str(c), ha="center", fontsize=11)
    axes[1].set_ylabel("Subjects")
    axes[1].set_title(f"(b) Sex  (M={m}, F={f})", fontsize=10)
    plt.tight_layout()
    plt.savefig(ASSETS / "demographics.png", bbox_inches="tight")
    plt.close()


# ── FIG 8 · Domain alignment (QT) ─────────────────────────────────────────────
def fig_domain_alignment():
    # reuse the pipeline-generated one if present else skip
    src = Path("results/hybrid_pipeline_v4/v4_domain_alignment.png")
    if src.exists():
        import shutil
        shutil.copy(src, ASSETS / "domain_alignment.png")


if __name__ == "__main__":
    fig_two_model();        print("  two_model_dose_response.png")
    fig_dose_response();    print("  dose_response.png")
    fig_roc();              print("  lopo_roc.png")
    fig_features();         print("  feature_analysis.png")
    fig_patient_scores();   print("  patient_scores.png")
    fig_charis_comparison();print("  charis_model_comparison.png")
    fig_demographics();     print("  demographics.png")
    fig_domain_alignment(); print("  domain_alignment.png")
    print("All README figures -> assets/")
