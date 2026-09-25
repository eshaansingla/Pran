"""Figures for hardware-vasalva-method: one results.png (+ feature_importance.png) per model folder and the comparison figures.

    python hardware-vasalva-method/plots.py

Reads only what run_all.py wrote (summary.csv, <Model>/metrics.json, per_fold.csv, predictions.csv, model.pkl); nothing is recomputed.
"Development" curves are out-of-fold (every subject scored by a model that never saw them); "final test" curves are the 29 untouched subjects.
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

HERE = Path(__file__).resolve().parent
import sys; sys.path.insert(0, str(HERE.parent))
from pran.features import FEATURE_NAMES  # noqa: E402  (run from the repo root)
INK, INK2, GRID, BLUE, ORANGE, SURF, AQUA = "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6", "#eb6834", "#fcfcfb", "#1baf7a"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
                     "axes.titleweight": "bold", "axes.titlelocation": "left"})


def model_figure(name):
    d = HERE / name; m = json.loads((d / "metrics.json").read_text()); f = pd.read_csv(d / "per_fold.csv"); pr = pd.read_csv(d / "predictions.csv")
    dv, te = pr[pr.set == "development"], pr[pr.set == "final_test"]; dm, tm, oc = m["dev_cv"], m["final_test"], m["overfit_check"]
    fig, axs = plt.subplots(2, 3, figsize=(16, 8.6))
    ax = axs[0, 0]
    for df, c, lab in [(dv, BLUE, f"development, out-of-fold (AUC {dm['auc']:.3f})"), (te, ORANGE, f"final test, 29 unseen subjects (AUC {tm['auc']:.3f})")]:
        fp, tp, _ = roc_curve(df.label, df.score); ax.plot(fp, tp, color=c, lw=2, label=lab)
    ax.plot([0, 1], [0, 1], color=INK2, ls="--", lw=1); ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate"); ax.set_title("ROC"); ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax = axs[0, 1]
    for df, c, lab in [(dv, BLUE, f"development (AP {dm['avg_precision']:.3f})"), (te, ORANGE, f"final test (AP {tm['avg_precision']:.3f})")]:
        p_, r_, _ = precision_recall_curve(df.label, df.score); ax.plot(r_, p_, color=c, lw=2, label=lab)
    ax.axhline(dm["avg_precision_chance"], color=INK2, ls="--", lw=1, label=f"chance ({dm['avg_precision_chance']:.3f})"); ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_title("Precision-recall")
    ax.legend(frameon=False, fontsize=8, loc="lower left"); ax.set_ylim(0, 1.02)
    ax = axs[0, 2]; cm = np.array([[tm["TN"], tm["FP"]], [tm["FN"], tm["TP"]]]); ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2): r = cm[i, j] / cm[i].sum(); ax.text(j, i, f"{cm[i, j]:,}\n{r:.1%} of row", ha="center", va="center", color="white" if r > .5 else INK)
    ax.set_xticks([0, 1], ["Predicted normal", "Predicted Valsalva"]); ax.set_yticks([0, 1], ["Actually normal", "Actually Valsalva"]); ax.set_title("Final test: confusion matrix")
    [s.set_visible(False) for s in ax.spines.values()]
    ax = axs[1, 0]; bins = np.linspace(te.score.min(), te.score.max(), 50)
    ax.hist(te.score[te.label == 0], bins, color=BLUE, alpha=.7, label="normal windows", density=True); ax.hist(te.score[te.label == 1], bins, color=ORANGE, alpha=.7, label="Valsalva windows", density=True)
    ax.axvline(tm["threshold"], color=INK, ls="--", lw=1, label=f"threshold {tm['threshold']:.2f}"); ax.set_xlabel("model score"); ax.set_ylabel("density"); ax.set_title("Final test: score distributions"); ax.legend(frameon=False, fontsize=8)
    ax = axs[1, 1]; x = np.arange(len(f)); ax.bar(x, f.auc, color=BLUE, width=.6, label="held-out AUC"); ax.plot(x, f.train_auc, "o", color=ORANGE, ms=6, label="training AUC")
    ax.set_xticks(x, [f"F{i + 1}" for i in x]); ax.set_ylim(min(0.5, f.auc.min() - .05), 1.02); ax.set_title(f"Development folds: train vs held-out AUC (gap {oc['gap']:+.3f}, {oc['verdict']})"); ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    ax = axs[1, 2]; ax.axis("off"); rows = [("", "Development CV", "Final test"), ("AUC", f"{dm['auc']:.3f}", f"{tm['auc']:.3f}"), ("Sensitivity", f"{dm['sensitivity']:.3f}", f"{tm['sensitivity']:.3f}"),
                                             ("Specificity", f"{dm['specificity']:.3f}", f"{tm['specificity']:.3f}"), ("Precision", f"{dm['precision']:.3f}", f"{tm['precision']:.3f}"), ("F1", f"{dm['f1']:.3f}", f"{tm['f1']:.3f}"),
                                             ("Balanced acc.", f"{dm['balanced_accuracy']:.3f}", f"{tm['balanced_accuracy']:.3f}"), ("MCC", f"{dm['mcc']:.3f}", f"{tm['mcc']:.3f}"),
                                             ("Valsalva top of 4", dm["valsalva_session_highest_of_4"], tm["valsalva_session_highest_of_4"])]
    tb = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center"); tb.auto_set_font_size(False); tb.set_fontsize(10); tb.scale(1, 1.6)
    for (r, c), cl in tb.get_celld().items(): cl.set_edgecolor(GRID); cl.set_facecolor(SURF); cl.set_text_props(fontweight="bold") if r == 0 else None
    fig.suptitle(f"{name}: hardware Valsalva (abnormal) vs other windows (normal), subject-grouped", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.945, f"Development AUC {dm['auc']:.3f} (95% CI {dm['auc_ci95_subject_bootstrap'][0]:.3f} to {dm['auc_ci95_subject_bootstrap'][1]:.3f}) | final test AUC {tm['auc']:.3f} "
             f"(95% CI {tm['auc_ci95_subject_bootstrap'][0]:.3f} to {tm['auc_ci95_subject_bootstrap'][1]:.3f}) | average precision chance level {dm['avg_precision_chance']:.3f}", color=INK2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(d / "results.png", dpi=130); plt.close(fig)
    mdl = pickle.load(open(d / "model.pkl", "rb")); fi = getattr(mdl, "feature_importances_", None)
    if fi is None and hasattr(mdl, "get_feature_importance"): fi = mdl.get_feature_importance()
    if fi is None and hasattr(mdl, "coef_"): fi = np.abs(np.ravel(mdl.coef_))
    if fi is not None:
        fi = np.asarray(fi, float); fi = 100 * fi / fi.sum(); o = np.argsort(fi)[-12:]
        fig, ax = plt.subplots(figsize=(8, 4.8)); ax.barh(np.array(FEATURE_NAMES)[o], fi[o], color=BLUE, height=.6)
        for y_, v in enumerate(fi[o]): ax.text(v + .2, y_, f"{v:.1f}%", va="center", color=INK2, fontsize=8.5)
        ax.set_xlim(0, fi[o].max() * 1.15); ax.set_xlabel("share of total importance (%)"); ax.set_title(f"{name}: top 12 features (model trained on development subjects)")
        ax.grid(axis="x", color=GRID); ax.set_axisbelow(True); fig.tight_layout(); fig.savefig(d / "feature_importance.png", dpi=130); plt.close(fig)


def comparison():
    S = pd.read_csv(HERE / "summary.csv"); best = S.model[0]; foot = "Winner chosen on development subjects only; final-test subjects were scored once."
    T = S.iloc[::-1]; y = np.arange(len(T)); fig, ax = plt.subplots(figsize=(9.5, 6.4))
    ax.barh(y, T.dev_auc, color=[ORANGE if m == best else BLUE for m in T.model], height=.62)
    ax.errorbar(T.dev_auc, y, xerr=[T.dev_auc - T.dev_auc_lo, T.dev_auc_hi - T.dev_auc], fmt="none", ecolor=INK2, lw=1, capsize=3)
    ax.plot(T.test_auc, y, "D", color=INK, ms=5, label="final test AUC (29 unseen subjects)")
    for i, (a, b) in enumerate(zip(T.dev_auc, T.test_auc)): ax.text(1.003, i, f"{a:.3f} | {b:.3f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, T.model); ax.set_xlim(0.6, 1.06); ax.set_xlabel("development AUC (bars, 95% CI over subjects)   |   final test AUC (diamonds)"); ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    ax.set_title(f"Hardware Valsalva vs other: {len(S)} models (orange = best on development)"); ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
    fig.text(0.01, 0.005, foot, fontsize=7.5, color=INK2); fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig(HERE / "ranking_auc.png", dpi=130); plt.close(fig)
    cols = [("dev_auc", "AUC"), ("dev_ap", "Avg precision"), ("dev_sens", "Sensitivity"), ("dev_spec", "Specificity"), ("dev_prec", "Precision"), ("dev_f1", "F1"), ("dev_bal_acc", "Balanced acc."), ("dev_mcc", "MCC")]
    A = T[[c for c, _ in cols]].values; fig, ax = plt.subplots(figsize=(10, 6.4)); ax.imshow(A, cmap="Blues", vmin=.3, vmax=1, aspect="auto")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]): ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", color="white" if A[i, j] > .7 else INK, fontsize=9)
    ax.set_xticks(range(len(cols)), [n for _, n in cols], rotation=20, ha="right"); ax.set_yticks(range(len(T)), T.model); ax.set_title("Development cross-validation metrics (out-of-fold, pooled)")
    [s.set_visible(False) for s in ax.spines.values()]; fig.text(0.01, 0.005, foot, fontsize=7.5, color=INK2); fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig(HERE / "metrics_heatmap.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9.5, 6.4)); ax.plot([.5, 1.02], [.5, 1.02], color=INK2, ls="--", lw=1); ax.plot([.5, 1.02], [.55, 1.07], color=GRID, lw=1)
    ax.scatter(T.train_auc, T.heldout_fold_auc, s=70, color=[ORANGE if m == best else BLUE for m in T.model], edgecolor=SURF, lw=1.5, zorder=3)
    for r in T.itertuples(): ax.annotate(r.model, (r.train_auc, r.heldout_fold_auc), xytext=(6, -3), textcoords="offset points", fontsize=8.5, color=INK2)
    ax.set_xlim(.75, 1.02); ax.set_ylim(.75, 1.02); ax.set_xlabel("training AUC (subjects the model was fit on)"); ax.set_ylabel("held-out AUC (subjects never seen)")
    ax.set_title("Over/under-fitting check: points on the dashed line = no gap; below it = overfit"); ax.grid(color=GRID); ax.set_axisbelow(True)
    fig.text(0.01, 0.005, "Light line marks a 0.05 gap (our overfit threshold). Low held-out AUC with low training AUC would indicate underfitting.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig(HERE / "overfit_underfit_check.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5, 6.5)); ax.scatter(T.dev_spec, T.dev_sens, s=70, color=[ORANGE if m == best else BLUE for m in T.model], edgecolor=SURF, lw=1.5, zorder=3)
    for r in T.itertuples(): ax.annotate(r.model, (r.dev_spec, r.dev_sens), xytext=(6, 4), textcoords="offset points", fontsize=8.5, color=INK2)
    ax.set_xlabel("specificity (development)"); ax.set_ylabel("sensitivity (development)"); ax.set_title("Sensitivity vs specificity (top right is better)"); ax.grid(color=GRID); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(HERE / "sensitivity_vs_specificity.png", dpi=130); plt.close(fig)
    t = T[["model", "dev_auc", "dev_sens", "dev_spec", "dev_prec", "dev_f1", "test_auc", "test_sens", "test_spec", "test_prec", "test_f1", "train_heldout_gap", "fit_verdict"]].copy()
    t.columns = ["Model", "Dev AUC", "Dev sens", "Dev spec", "Dev prec", "Dev F1", "Test AUC", "Test sens", "Test spec", "Test prec", "Test F1", "Train-held gap", "Fit"]
    cell = [[r[0]] + [f"{v:.3f}" if isinstance(v, float) else str(v) for v in r[1:]] for r in t.values.tolist()][::-1]
    fig, ax = plt.subplots(figsize=(15, .42 * len(t) + 1.4)); ax.axis("off"); tb = ax.table(cellText=cell, colLabels=list(t.columns), loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.auto_set_column_width(list(range(len(t.columns)))); tb.scale(1, 1.5)
    for (r, c), cl in tb.get_celld().items():
        cl.set_edgecolor(GRID); cl.set_facecolor(SURF)
        if r == 0: cl.set_text_props(fontweight="bold")
        if r == 1: cl.set_facecolor("#fbe3d8")
    ax.set_title("End results, ranked by development AUC (best highlighted)", pad=6); fig.text(0.01, 0.005, foot, fontsize=7.5, color=INK2); fig.tight_layout(rect=(0, .03, 1, 1))
    fig.savefig(HERE / "end_results_table.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    S = pd.read_csv(HERE / "summary.csv")
    for n in S.model: model_figure(n)
    comparison(); print("figures written")
