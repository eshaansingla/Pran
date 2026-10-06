"""Figures for charis/compare_models.py output: one results.png per model folder + comparison figures.

    python charis/plot_compare.py

Reads models/charis_compare/ (summary.csv, <Model>/per_patient.csv, <Model>/model.pkl) and writes PNGs next to them,
and copies the best model's figures into models/charis_best/.
All per-patient numbers are leave-one-patient-out (test patient never seen in training). Feature-importance figures come
from the final model trained on all 13 patients, so they describe the model, not held-out performance.
"""
from __future__ import annotations
import pickle
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT, BEST = Path("models/charis_compare"), Path("models/charis_best")
FEATURES = ["cardiac_amplitude", "cardiac_frequency", "respiratory_amplitude", "slow_wave_power", "cardiac_power"]
INK, INK2, GRID, BLUE, ORANGE, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6", "#eb6834", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
                     "axes.titleweight": "bold", "axes.titlelocation": "left"})
# Comparison figures cover the originally requested model set. HistGradBoost and MLP were also trained/evaluated
# (their folders and rows in summary.csv are kept); every comparison figure carries this footnote.
EXCLUDE = {"HistGradBoost", "MLP_64_32"}
FOOT = "Also evaluated separately (not shown): HistGradBoost AUC 0.957, MLP AUC 0.955; not significantly different from XGBoost (see summary.csv)."


def _foot(fig):
    fig.text(0.01, 0.005, FOOT, color=INK2, fontsize=7.5, ha="left", va="bottom")


METRICS = [("auc_mean", "AUC"), ("sensitivity", "Sensitivity"), ("specificity", "Specificity"), ("precision", "Precision"),
           ("f1", "F1"), ("avg_precision", "Average precision")]


def _grid(ax):
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)


def per_patient_bar(ax, d, col, title, flag=None):
    x = np.arange(len(d)); v = d[col].values
    bad = v < flag if flag is not None else np.zeros(len(v), bool)
    ax.bar(x, v, color=[ORANGE if b else BLUE for b in bad], width=0.7)
    ax.axhline(np.mean(v), color=INK2, lw=1, ls="--"); ax.text(len(d) - 0.4, np.mean(v) + 0.02, f"mean {np.mean(v):.3f}", ha="right", color=INK2, fontsize=8, bbox=dict(fc=SURF, ec="none", pad=1.5))
    ax.set_xticks(x); ax.set_xticklabels([f"P{p}" for p in d.patient], fontsize=8); ax.set_ylim(0, 1.08); ax.set_title(title); _grid(ax)


def model_figure(name, d, s, out):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.2))
    per_patient_bar(axs[0, 0], d, "auc", "AUC per patient (orange = below 0.90)", 0.90)
    per_patient_bar(axs[0, 1], d, "sensitivity", "Sensitivity per patient")
    per_patient_bar(axs[0, 2], d, "specificity", "Specificity per patient")
    per_patient_bar(axs[1, 0], d, "precision", "Precision per patient")
    per_patient_bar(axs[1, 1], d, "f1", "F1 per patient")
    ax = axs[1, 2]; tn, fp, fn, tp = (int(d[c].sum()) for c in ("TN", "FP", "FN", "TP")); cm = np.array([[tn, fp], [fn, tp]])
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            r = cm[i, j] / cm[i].sum(); ax.text(j, i, f"{cm[i, j]:,}\n{r:.1%} of row", ha="center", va="center", color="white" if r > 0.5 else INK, fontsize=10)
    ax.set_xticks([0, 1], ["Predicted normal", "Predicted elevated"]); ax.set_yticks([0, 1], ["Actually normal", "Actually elevated"])
    ax.set_title("All 13 patients pooled: confusion matrix"); [sp.set_visible(False) for sp in ax.spines.values()]
    fig.suptitle(f"{name}: CHARIS-only, leave-one-patient-out (13 patients)", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.94, f"AUC {s.auc_mean:.3f} (95% CI {s.auc_ci_lo:.3f} to {s.auc_ci_hi:.3f})   sensitivity {s.sensitivity:.3f}   specificity {s.specificity:.3f}   "
             f"precision {s.precision:.3f}   F1 {s.f1:.3f}   average precision {s.avg_precision:.3f}   patients with AUC >= 0.90: {int(s.patients_auc_ge_0_90)}/13",
             color=INK2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(out / "results.png", dpi=140); plt.close(fig)


def importance_figure(name, out):
    m = pickle.load(open(out / "model.pkl", "rb"))
    fi = getattr(m, "feature_importances_", None)
    if fi is None and hasattr(m, "get_feature_importance"): fi = m.get_feature_importance()
    if fi is None and hasattr(m, "coef_"): fi = np.abs(np.ravel(m.coef_))
    if fi is None: return
    fi = np.asarray(fi, float); fi = 100 * fi / fi.sum(); o = np.argsort(fi)
    fig, ax = plt.subplots(figsize=(7, 3.6)); ax.barh(np.array(FEATURES)[o], fi[o], color=BLUE, height=0.6)
    for y, v in enumerate(fi[o]): ax.text(v + 0.5, y, f"{v:.1f}%", va="center", color=INK2, fontsize=9)
    ax.set_xlim(0, max(fi) * 1.15); ax.set_title(f"{name}: feature importance (final model, all 13 patients)"); ax.set_xlabel("share of total importance (%)")
    ax.grid(axis="x", color=GRID); ax.set_axisbelow(True); fig.tight_layout(); fig.savefig(out / "feature_importance.png", dpi=140); plt.close(fig)


def comparison_figures(S, D):
    S = S.sort_values("auc_mean"); best = S.model.iloc[-1]
    fig, ax = plt.subplots(figsize=(9, 6)); y = np.arange(len(S))
    ax.barh(y, S.auc_mean, color=[ORANGE if m == best else BLUE for m in S.model], height=0.62)
    ax.errorbar(S.auc_mean, y, xerr=[S.auc_mean - S.auc_ci_lo, S.auc_ci_hi - S.auc_mean], fmt="none", ecolor=INK2, lw=1, capsize=3)
    for i, v in enumerate(S.auc_mean): ax.text(1.005, i, f"{v:.3f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, S.model); ax.set_xlim(0.5, 1.04); ax.set_xlabel("mean leave-one-patient-out AUC (bars: 95% CI over patients)")
    ax.set_title(f"CHARIS-only: {len(S)} models ranked by AUC (orange = top of this set)"); ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); _foot(fig); fig.savefig(ROOT / "ranking_auc.png", dpi=140); plt.close(fig)

    cols = [c for c, _ in METRICS]; T = S.iloc[::-1].set_index("model")[cols].values
    fig, ax = plt.subplots(figsize=(9, 6.4)); im = ax.imshow(T, cmap="Blues", vmin=0.3, vmax=1, aspect="auto")
    for i in range(T.shape[0]):
        for j in range(T.shape[1]): ax.text(j, i, f"{T[i, j]:.2f}", ha="center", va="center", color="white" if T[i, j] > 0.7 else INK, fontsize=9)
    ax.set_xticks(range(len(cols)), [n for _, n in METRICS], rotation=20, ha="right"); ax.set_yticks(range(len(S)), S.iloc[::-1].model)
    ax.set_title("All metrics, all models (mean over 13 patients)"); [sp.set_visible(False) for sp in ax.spines.values()]
    fig.tight_layout(rect=(0, 0.03, 1, 1)); _foot(fig); fig.savefig(ROOT / "metrics_heatmap.png", dpi=140); plt.close(fig)

    P = D.pivot(index="model", columns="patient", values="auc").loc[S.iloc[::-1].model]
    fig, ax = plt.subplots(figsize=(11, 6.4)); ax.imshow(P.values, cmap="Blues", vmin=0.6, vmax=1, aspect="auto")
    for i in range(P.shape[0]):
        for j in range(P.shape[1]): ax.text(j, i, f"{P.values[i, j]:.2f}", ha="center", va="center", color="white" if P.values[i, j] > 0.85 else INK, fontsize=8)
    ax.set_xticks(range(P.shape[1]), [f"P{c}" for c in P.columns]); ax.set_yticks(range(P.shape[0]), P.index)
    ax.set_title("AUC per patient and model (Patient 4 is hard for every model)"); [sp.set_visible(False) for sp in ax.spines.values()]
    fig.tight_layout(rect=(0, 0.03, 1, 1)); _foot(fig); fig.savefig(ROOT / "patient_auc_heatmap.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6.5)); ax.scatter(S.specificity, S.sensitivity, s=70, color=[ORANGE if m == best else BLUE for m in S.model], edgecolor=SURF, lw=1.5, zorder=3)
    for _, r in S.iterrows(): ax.annotate(r.model, (r.specificity, r.sensitivity), xytext=(6, 4), textcoords="offset points", fontsize=8.5, color=INK2)
    ax.set_xlabel("specificity (mean)"); ax.set_ylabel("sensitivity (mean)"); ax.set_title("Sensitivity vs specificity by model (top right is better)")
    ax.grid(color=GRID); ax.set_axisbelow(True); fig.tight_layout(rect=(0, 0.03, 1, 1)); _foot(fig); fig.savefig(ROOT / "sensitivity_vs_specificity.png", dpi=140); plt.close(fig)

    t = S.iloc[::-1][["model", "auc_mean", "auc_std", "auc_worst_patient", "patients_auc_ge_0_90", "sensitivity", "specificity", "precision", "f1"]].copy()
    t.columns = ["Model", "AUC", "AUC SD", "Worst pt", "Pts >=0.90", "Sens", "Spec", "Prec", "F1"]
    cell = [[r.Model] + [f"{v:.3f}" if isinstance(v, float) else str(int(v)) for v in r.values[1:]] for _, r in t.iterrows()]
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(t) + 1.2)); ax.axis("off")
    tb = ax.table(cellText=cell, colLabels=list(t.columns), loc="center", cellLoc="center"); tb.auto_set_font_size(False); tb.set_fontsize(9.5); tb.auto_set_column_width(list(range(len(t.columns)))); tb.scale(1, 1.5)
    for (r, c), cl in tb.get_celld().items():
        cl.set_edgecolor(GRID); cl.set_facecolor(SURF)
        if r == 0: cl.set_text_props(fontweight="bold")
        if r == 1: cl.set_facecolor("#fbe3d8")
        if c == 0: cl.set_text_props(ha="left")
    ax.set_title("End results, ranked by mean AUC (best highlighted)", pad=6); fig.tight_layout(rect=(0, 0.03, 1, 1)); _foot(fig); fig.savefig(ROOT / "end_results_table.png", dpi=140); plt.close(fig)


def main():
    S = pd.read_csv(ROOT / "summary.csv"); D = pd.concat([pd.read_csv(ROOT / m / "per_patient.csv") for m in S.model], ignore_index=True)
    for _, s in S.iterrows():
        out = ROOT / s.model; model_figure(s.model, D[D.model == s.model].sort_values("patient"), s, out); importance_figure(s.model, out)
    Ssub = S[~S.model.isin(EXCLUDE)].reset_index(drop=True)
    comparison_figures(Ssub, D[D.model.isin(Ssub.model)])
    best = Ssub.sort_values("auc_mean").model.iloc[-1]
    for f in ("results.png", "feature_importance.png"):
        if (ROOT / best / f).exists(): shutil.copy(ROOT / best / f, BEST / f)
    (BEST / "README.txt").write_text(f"Chosen CHARIS-only model: {best} (top by mean leave-one-patient-out AUC among the 12 models in the comparison figures).\n"
                                     f"{FOOT}\nInput: 5 features via qt_scaler.pkl. Output: P(window has ICP >= 20 mmHg). Threshold in metrics.json.\n")
    for f in ("ranking_auc.png", "metrics_heatmap.png", "patient_auc_heatmap.png", "sensitivity_vs_specificity.png", "end_results_table.png"): shutil.copy(ROOT / f, BEST / f)
    print("figures written; best =", best)


if __name__ == "__main__":
    main()
