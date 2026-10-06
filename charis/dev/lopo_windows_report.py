"""Per-patient leave-one-patient-out window counts (flagged / not flagged) for the CHARIS XGBoost model.

    python charis/lopo_windows_report.py

Reads models/charis_compare/XGBoost/per_patient.csv (written by compare_models.py) and writes
results/charis_lopo/patient_XX.png (one per held-out patient), overview.png and lopo_windows.csv.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "models" / "charis_compare" / "XGBoost" / "per_patient.csv"
OUT = ROOT / "results" / "charis_lopo"
BLUE, RED, GREY = "#2f6bff", "#ff5a3c", "#8b91a1"


def patient_fig(r):
    tn, fp, fn, tp = (int(r[k]) for k in ("TN", "FP", "FN", "TP"))
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw=dict(width_ratios=[1, 1.1, 1]))
    cm = np.array([[tn, fp], [fn, tp]]); rate = cm / cm.sum(1, keepdims=True)
    a.imshow(rate, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{cm[i, j]:,}\n{rate[i, j]:.0%}", ha="center", va="center", fontsize=13, fontweight="bold",
                   color="white" if rate[i, j] > .5 else "black")
    a.set_xticks([0, 1], ["not flagged", "flagged"]); a.set_yticks([0, 1], ["truly normal\n(ICP < 20)", "truly elevated\n(ICP >= 20)"])
    a.set_title("Model call vs ground truth (10 s windows)", fontsize=11)
    x = np.arange(2)
    b.bar(x - .2, [tn + fn, fp + tp], .4, color=GREY, label="model: not flagged | flagged")
    b.bar(x + .2, [tn + fp, tp + fn], .4, color=[BLUE, RED], label="truth: normal | elevated")
    for xi, v in zip(np.r_[x - .2, x + .2], [tn + fn, fp + tp, tn + fp, tp + fn]):
        b.text(xi, v, f"{v:,}", ha="center", va="bottom", fontsize=10)
    b.set_xticks(x, ["normal windows\n/ not flagged", "elevated windows\n/ flagged"]); b.set_ylabel("windows")
    b.set_ylim(0, max(tn + fn, fp + tp, tn + fp, tp + fn) * 1.3); b.legend(frameon=False, fontsize=9, loc="upper center"); b.spines[["top", "right"]].set_visible(False)
    b.set_title(f"{r.n_windows:,.0f} windows, {r.abnormal_pct:.1f}% truly elevated", fontsize=11)
    if "train_auc" in r and pd.notna(r.train_auc):
        m = np.arange(3); w = .26
        for k, (tag, col) in enumerate((("train", GREY), ("val", "#9db9ff"), ("test", BLUE))):
            v = [r[f"{tag}_auc"], r[f"{tag}_sens"], r[f"{tag}_spec"]]
            c.bar(m + (k - 1) * w, v, w, color=col, label=f"{tag} (n={int(r[tag + '_n']):,})")
            for xi, vi in zip(m + (k - 1) * w, v): c.text(xi, vi, f"{vi:.2f}", ha="center", va="bottom", fontsize=8)
        c.set_xticks(m, ["AUC", "sensitivity", "specificity"]); c.set_ylim(0, 1.25); c.legend(frameon=False, fontsize=8, ncol=3, loc="upper center")
        c.spines[["top", "right"]].set_visible(False); c.set_title("Training vs held-out test (same threshold)", fontsize=11)
    else:
        c.axis("off")
    fig.suptitle(f"CHARIS patient {int(r.patient)} held out  |  AUC {r.auc:.3f}  sensitivity {r.sensitivity:.1%}  "
                 f"specificity {r.specificity:.1%}  precision {r.precision:.1%}", fontsize=12, fontweight="bold")
    fig.text(.5, .01, "Model trained on the other 12 patients only. Leave-one-patient-out, XGBoost. "
             "Mean-level pressure information is still in these features (clean AUC ~0.65-0.69).", ha="center", fontsize=8, color=GREY)
    fig.tight_layout(rect=(0, .03, 1, .94)); fig.savefig(OUT / f"patient_{int(r.patient):02d}.png", dpi=130); plt.close(fig)


def overview(d):
    fig, ax = plt.subplots(figsize=(12, 5)); x = np.arange(len(d))
    tot = d.n_windows.values
    for off, (lo, hi, cs, lab) in {-.2: (d.TN, d.FP, (BLUE, "#9db9ff"), "truly normal"), .2: (d.TP, d.FN, (RED, "#ffb4a3"), "truly elevated")}.items():
        ax.bar(x + off, lo / tot * 100, .38, color=cs[0], label=f"{lab}: correct call")
        ax.bar(x + off, hi / tot * 100, .38, bottom=lo / tot * 100, color=cs[1], label=f"{lab}: wrong call")
    ax.set_xticks(x, [f"P{int(p)}\nAUC {a:.2f}" for p, a in zip(d.patient, d.auc)], fontsize=8)
    ax.set_ylabel("% of that patient's windows"); ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("LOPO: windows flagged / not flagged per held-out CHARIS patient", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "overview.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(SRC).sort_values("patient").reset_index(drop=True)
    if (OUT / "train_test.csv").exists():
        d = d.merge(pd.read_csv(OUT / "train_test.csv").drop(columns=["threshold"]), on="patient", how="left")
    d["flagged"] = d.TP + d.FP; d["not_flagged"] = d.TN + d.FN
    d[["patient", "n_windows", "abnormal_pct", "flagged", "not_flagged", "TP", "FP", "TN", "FN", "auc", "sensitivity", "specificity", "precision"] + [c for c in d.columns if c.startswith(("train_", "val_", "test_"))]].to_csv(OUT / "lopo_windows.csv", index=False)
    for _, r in d.iterrows():
        patient_fig(r)
    overview(d)
    print(f"wrote {len(d)} patient images + overview.png + lopo_windows.csv to {OUT}")
