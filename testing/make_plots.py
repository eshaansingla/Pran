"""Figures for testing/ from the saved A_* and B_* result files.  python testing/make_plots.py"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

D = Path(__file__).parent; BLUE, ORANGE, INK, GRID = "#2a78d6", "#eb6834", "#2b2a28", "#e6e5e1"
A = json.loads((D / "A_charis_model_on_hw.json").read_text()); B = json.loads((D / "B_hw_model_on_charis.json").read_text())
sa = pd.read_csv(D / "A_charis_model_on_hw_per_subject.csv"); sb = pd.read_csv(D / "B_hw_model_on_charis_per_patient.csv")
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK})
def clean(ax): ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)

# ---- Figure A
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6)); fs = A["flagged_pct_by_session"]
b = ax[0].bar(list(fs), list(fs.values()), color=[BLUE] * 3 + [ORANGE], width=.6)
for r, v in zip(b, fs.values()): ax[0].text(r.get_x() + r.get_width() / 2, v + .2, f"{v:.1f}%", ha="center")
ax[0].set(title="Windows flagged by CHARIS model, by session", ylabel="% windows flagged"); clean(ax[0])
ax[1].hist(sa.auc_valsalva_vs_rest.dropna(), bins=25, color=BLUE, edgecolor="white"); ax[1].axvline(.5, color=INK, ls="--", lw=1); ax[1].axvline(sa.auc_valsalva_vs_rest.mean(), color=ORANGE, lw=2)
ax[1].text(.5, ax[1].get_ylim()[1] * .95, " chance", va="top"); ax[1].text(sa.auc_valsalva_vs_rest.mean(), ax[1].get_ylim()[1] * .8, f" mean {sa.auc_valsalva_vs_rest.mean():.2f}", color=ORANGE, va="top")
ax[1].set(title="Within-subject AUC, Valsalva vs rest (146 subjects)", xlabel="AUC", ylabel="subjects"); clean(ax[1])
ag = [("age < 30", A["flagged_pct_age_under_30"]), ("age > 65", A["flagged_pct_age_over_65"])]
b = ax[2].bar([g[0] for g in ag], [g[1] for g in ag], color=[BLUE, ORANGE], width=.5)
for r, (_, v) in zip(b, ag): ax[2].text(r.get_x() + r.get_width() / 2, v + .4, f"{v:.1f}%", ha="center")
ax[2].set(title=f"Age confound (AUC over-65 vs under-30 = {A['age_confound_check_auc_over65_vs_under30']:.2f})", ylabel="% windows flagged"); clean(ax[2])
fig.suptitle(f"Test A: CHARIS model on hardware. Pooled AUC {A['pooled_auc_valsalva_vs_rest']:.3f}, sens {A['sensitivity']:.2f}, spec {A['specificity']:.2f}", x=.01, ha="left", fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(D / "A_charis_model_on_hw.png", dpi=140); plt.close(fig)

# ---- Figure B
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [2.2, 1]}); s = sb.sort_values("patient")
ax[0].bar([f"P{p}" for p in s.patient], s.auc, color=[BLUE if a > .5 else ORANGE for a in s.auc], width=.6)
for i, a in enumerate(s.auc): ax[0].text(i, a + .01, f"{a:.2f}", ha="center", fontsize=9)
ax[0].axhline(.5, color=INK, ls="--", lw=1); ax[0].text(len(s) - .5, .51, "chance", ha="right", fontsize=9); ax[0].set(ylim=(0, 1), ylabel="AUC", title="Per-patient AUC on CHARIS (blue > chance, orange < chance)"); clean(ax[0])
m = [("hardware\nheld-out", B["hw_own_heldout_auc"]), ("CHARIS\npooled", B["pooled_auc"]), ("CHARIS mean\nper patient", B["mean_per_patient_auc"])]
b = ax[1].bar([x[0] for x in m], [x[1] for x in m], color=[BLUE, ORANGE, ORANGE], width=.55)
for r, (_, v) in zip(b, m): ax[1].text(r.get_x() + r.get_width() / 2, v + .01, f"{v:.3f}", ha="center")
ax[1].axhline(.5, color=INK, ls="--", lw=1); ax[1].set(ylim=(0, 1), title="Summary AUC"); clean(ax[1])
fig.suptitle(f"Test B: hardware 5-feature model on CHARIS (ICP >= 20). sens {B['sensitivity']:.2f}, spec {B['specificity']:.2f}, balanced acc {B['balanced_accuracy']:.2f}", x=.01, ha="left", fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(D / "B_hw_model_on_charis.png", dpi=140); plt.close(fig)

# ---- metrics table
rows = [["Metric", "A: CHARIS model -> hardware", "B: hardware model -> CHARIS"],
        ["Task", "Valsalva vs rest", "ICP >= 20 mmHg"],
        ["Pooled AUC", f"{A['pooled_auc_valsalva_vs_rest']:.3f}", f"{B['pooled_auc']:.3f}"],
        ["Mean per-subject/patient AUC", f"{A['mean_within_subject_auc_vs_rest']:.3f}", f"{B['mean_per_patient_auc']:.3f}"],
        ["Sensitivity", f"{A['sensitivity']:.3f}", f"{B['sensitivity']:.3f}"],
        ["Specificity", f"{A['specificity']:.3f}", f"{B['specificity']:.3f}"],
        ["Units", f"{A['n_subjects']} subjects, {A['n_windows']:,} windows", f"{B['charis_patients']} patients, {B['charis_windows']:,} windows"]]
fig, ax = plt.subplots(figsize=(10, 3)); ax.axis("off"); t = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center"); t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 1.6)
for (r, c), cell in t.get_celld().items():
    cell.set_edgecolor(GRID)
    if r == 0: cell.set_facecolor("#f3f2ef"); cell.set_text_props(fontweight="bold")
fig.tight_layout(); fig.savefig(D / "cross_test_metrics.png", dpi=140); plt.close(fig)
