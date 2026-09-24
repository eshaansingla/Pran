"""app.py - Pran web app: upload a hardware recording, see what the trained model recognises, and read the validated results.

    python app.py        # opens http://127.0.0.1:5000

Pages: Analyse (upload / example) | Results (held-out metrics and validity checks) | About (data, method, limits).
The model recognises the recorded *state* (Valsalva / posture); it is not an ICP measurement or a diagnosis.
"""
from __future__ import annotations

import base64
import io
import json
import threading
import webbrowser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import predict
from pran.features import SESSION_NAMES

UPLOAD_DIR = Path("results/_uploads")
META_PATH = Path("models/hardware/meta.json")
CHARIS_RESULTS = Path("results/qt_pipeline/qt_results.json")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024

# Validity checks and comparisons recorded in support/code (hw_skeptic_checks.py, hw_validity_controls.py,
# hw_motion_check.py, hw_model_compare.py, align_*.py, final_models.py). Shown on the Results page.
VALIDITY = [
    ("Subjects shared between train and test", "None, in every fold (asserted in code)"),
    ("Label-shuffle null (labels shuffled inside each subject)", "AUC 0.497, i.e. chance"),
    ("Train vs held-out gap", "AUC 0.999 vs 0.995 (+0.004): no sign of overfitting"),
    ("Hold out contiguous subject-ID blocks (recording batches)", "AUC 0.992, worst block 0.981"),
    ("Train on early IDs, test on late IDs (and reverse)", "AUC 0.992 / 0.988"),
    ("Slow drift: first vs second half of a single session", "AUC 0.52 to 0.62 (little drift)"),
    ("Recording order vs ICP order (supine vs head-up)", "Head-up scored below supine in 100% of subjects; recording order predicts the opposite"),
    ("Body movement: motion sensor alone", "AUC 0.695; optical AUC stays 0.995 in each subject's quietest 25% of windows"),
    ("Relative band powers only (no amplitude features)", "AUC 0.958"),
    ("One channel at a time (band powers only)", "IR 0.914, red 0.910, displacement 0.720"),
]
MODEL_FAMILIES = [
    ("XGBoost (used)", "0.995", "0.925"), ("HistGradientBoosting", "0.994", "0.919"), ("MLP (64-32)", "0.993", "0.921"),
    ("Random Forest", "0.990", "0.896"), ("Extra Trees", "0.990", "0.895"), ("Logistic Regression", "0.978", "0.854"),
]
TRANSFER = [
    ("CHARIS model applied to hardware as first built", "0.656"), ("Unit-free features", "0.648"), ("Per-subject rank alignment", "0.587"),
    ("Rank alignment", "0.572"), ("Unit change (x292 on amplitude)", "0.560"), ("Joint distribution mapping", "0.529"),
]

INK, INK2, TEAL, CORAL, GRID = "#1f2a30", "#5b6b73", "#1b7f8c", "#e4572e", "#e6eef0"
BANDS = {1: "#eaf3f4", 0: "#f4f4f2", 2: "#fdeee8", 3: "#fbe0d6"}


def _plot(r: dict) -> str:
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [2.3, 1]}, facecolor="white")
    t = [(i * predict.STEP_SECONDS) / 60 for i in range(r["n_windows"])]
    if r.get("sessions") is not None:
        sess = r["sessions"]; start = 0
        for i in range(1, len(sess) + 1):
            if i == len(sess) or sess[i] != sess[start]:
                a.axvspan(t[start], t[i - 1] + 0.1, color=BANDS[sess[start]], lw=0)
                a.text((t[start] + t[i - 1]) / 2, 1.04, SESSION_NAMES[sess[start]], ha="center", va="bottom", fontsize=8.5, color=INK2)
                start = i
    a.plot(t, r["valsalva"], color=TEAL, lw=1.3, label="window score"); a.plot(t, r["valsalva_35s"], color=INK, lw=1.6, label="35 s average")
    a.axhline(r["threshold"], color=CORAL, lw=1.2, ls=(0, (4, 3))); a.text(t[-1], r["threshold"] + 0.02, "threshold", color=CORAL, ha="right", fontsize=8.5)
    a.set_ylim(-0.02, 1.02); a.set_xlabel("time in recording (min)"); a.set_ylabel("Valsalva-like score"); a.grid(axis="y", color=GRID)
    a.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), fontsize=8, frameon=False)
    for s in ("top", "right"): a.spines[s].set_visible(False); b.spines[s].set_visible(False)
    if r.get("per_session"):
        present = r["present"]; vals = [r["per_session"][s]["pct_flagged"] for s in present]
        b.bar(range(len(present)), vals, color=[CORAL if s == 3 else TEAL for s in present], width=0.6)
        b.set_xticks(range(len(present))); b.set_xticklabels([SESSION_NAMES[s].replace(" ", "\n") for s in present], fontsize=8.5)
        for i, v in enumerate(vals): b.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=9)
        b.set_ylim(0, 112); b.set_ylabel("windows flagged (%)"); b.grid(axis="y", color=GRID); b.set_axisbelow(True)
    else:
        b.axis("off")
    plt.tight_layout(); buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=140); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _examples() -> list[str]:
    return sorted((p.name for p in Path("hw-tests").glob("icp_*.csv")), key=lambda n: int(n.split("_")[1]))[:8] if Path("hw-tests").exists() else []


def _model_ready() -> bool:
    return META_PATH.exists()


@app.route("/")
def index():
    return render_template("index.html", examples=_examples(), ready=_model_ready(), page="analyse")


@app.route("/analyze", methods=["POST"])
def analyze():
    if not _model_ready():
        return render_template("result.html", r={"ok": False, "error": "The hardware model is not trained yet. Run: python -m pran.hardware_model"}, page="analyse")
    example = request.form.get("example")
    if example:
        path = Path("hw-tests") / secure_filename(example)
    else:
        f = request.files.get("file")
        if not f or not f.filename:
            return render_template("result.html", r={"ok": False, "error": "No file selected."}, page="analyse")
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True); path = UPLOAD_DIR / secure_filename(f.filename); f.save(path)
    try:
        r = predict.analyse_csv(path)
    except Exception as e:  # unreadable / malformed CSV
        r = {"ok": False, "error": f"Could not read this file: {e}"}
    plot = _plot(r) if r["ok"] else None
    return render_template("result.html", r=r, plot=plot, names=SESSION_NAMES, page="analyse")


@app.route("/results")
def results():
    meta = json.loads(META_PATH.read_text()) if _model_ready() else None
    charis = json.loads(CHARIS_RESULTS.read_text()).get("main_split") if CHARIS_RESULTS.exists() else None
    return render_template("results.html", meta=meta, charis=charis, validity=VALIDITY, families=MODEL_FAMILIES, transfer=TRANSFER, page="results")


@app.route("/about")
def about():
    return render_template("about.html", page="about")


if __name__ == "__main__":
    if _model_ready():
        predict.load_model()
    threading.Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(host="127.0.0.1", port=5000, debug=False)
