"""
app.py  -  Pran ICP-modulation demo (local web app)
===================================================
A tiny local site to DEMONSTRATE the model on a recording: upload a hardware CSV,
get the within-subject ICP-modulation report + a plot. No training happens - it
loads the already-trained zero-shot CHARIS model once and reuses it.

Run everything with a single command:
    python app.py
Then open http://127.0.0.1:5000  (it opens automatically).

Honest scope: this shows a RELATIVE ICP-modulation trend, not a diagnosis and not
a calibrated mmHg value. See the banner in the UI.
"""
from __future__ import annotations

import base64
import io
import threading
import webbrowser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template_string, request

import predict  # reuses the clean inference API + the already-trained model

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB upload cap
UPLOAD_DIR = Path("results/_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

SESSION_NAME = predict.SESSION_NAME
ORDER = predict.ORDER

# ── plot ─────────────────────────────────────────────────────────────────────
def make_plot(r: dict) -> str:
    """Return a base64 PNG of the within-subject ICP ladder (score + slow-wave)."""
    present = r["sessions_present"]
    labels = [SESSION_NAME[s] for s in present]
    scores = [r["per_session"][s] for s in present]
    slow = [r["slow_wave"][s] for s in present]

    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    x = range(len(present))
    ax1.bar(x, scores, color="#2d6cdf", alpha=0.85, width=0.6, label="ICP-elevation score")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("Relative ICP-elevation score", color="#2d6cdf")
    ax1.tick_params(axis="y", labelcolor="#2d6cdf")
    ax1.set_ylim(0, max(scores) * 1.25 + 1e-6)

    ax2 = ax1.twinx()
    ax2.plot(x, slow, color="#e8590c", marker="o", lw=2, label="slow-wave power (ICP biomarker)")
    ax2.set_ylabel("slow-wave power", color="#e8590c")
    ax2.tick_params(axis="y", labelcolor="#e8590c")

    title = "Within-subject ICP modulation across the manoeuvre ladder"
    if r["rho"] is not None:
        title += f"   (Spearman ρ = {r['rho']:+.2f})"
    ax1.set_title(title, fontsize=11)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ── templates ────────────────────────────────────────────────────────────────
BASE_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
       background: #f5f6f8; color: #1b1f24; }
.wrap { max-width: 820px; margin: 0 auto; padding: 28px 20px 60px; }
h1 { font-size: 1.5rem; margin: 0 0 4px; }
.sub { color: #5c6672; margin: 0 0 22px; font-size: .95rem; }
.banner { background: #fff4e5; border: 1px solid #ffd8a8; color: #8a4b00;
          padding: 12px 16px; border-radius: 10px; font-size: .88rem; margin-bottom: 22px; }
.card { background: #fff; border: 1px solid #e6e8eb; border-radius: 14px;
        padding: 22px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,.04); }
label.drop { display: block; border: 2px dashed #b8c0cc; border-radius: 12px;
             padding: 30px; text-align: center; cursor: pointer; color: #5c6672; }
label.drop:hover { border-color: #2d6cdf; color: #2d6cdf; }
input[type=file] { margin-top: 12px; }
button { background: #2d6cdf; color: #fff; border: 0; border-radius: 10px;
         padding: 11px 22px; font-size: 1rem; cursor: pointer; margin-top: 14px; }
button:hover { background: #1f57bd; }
table { width: 100%; border-collapse: collapse; margin-top: 8px; }
th, td { text-align: left; padding: 9px 10px; border-bottom: 1px solid #eef0f2; font-size: .95rem; }
th { color: #5c6672; font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.verdict { font-size: 1.05rem; font-weight: 600; padding: 14px 16px; border-radius: 10px; }
.ok { background: #e7f7ec; color: #1b7a3d; border: 1px solid #b6e6c6; }
.weak { background: #fdecea; color: #a11a1a; border: 1px solid #f3c2bd; }
.kv { display: flex; gap: 26px; flex-wrap: wrap; margin: 6px 0 2px; }
.kv div span { display: block; }
.kv .k { color: #5c6672; font-size: .8rem; }
.kv .v { font-size: 1.25rem; font-weight: 600; }
img { width: 100%; border-radius: 10px; margin-top: 8px; }
.err { background: #fdecea; color: #a11a1a; border: 1px solid #f3c2bd;
       padding: 14px 16px; border-radius: 10px; }
a.back { color: #2d6cdf; text-decoration: none; font-size: .95rem; }
.foot { color: #8a929c; font-size: .82rem; margin-top: 30px; }
nav { display: flex; gap: 6px; margin-bottom: 20px; }
nav a { padding: 9px 16px; border-radius: 9px; text-decoration: none; color: #5c6672;
        font-size: .92rem; font-weight: 600; background: #eef0f3; }
nav a.active { background: #2d6cdf; color: #fff; }
.flagbox { display: flex; align-items: center; gap: 18px; flex-wrap: wrap; }
.badge { font-size: 1.05rem; font-weight: 700; padding: 10px 18px; border-radius: 999px; }
.badge.normal { background: #e7f7ec; color: #1b7a3d; border: 1px solid #b6e6c6; }
.badge.flag { background: #fdecea; color: #a11a1a; border: 1px solid #f3c2bd; }
.meter { flex: 1; min-width: 220px; }
.meter .bar { height: 14px; background: #eef0f3; border-radius: 999px; overflow: hidden; position: relative; }
.meter .fill { height: 100%; background: linear-gradient(90deg,#2d6cdf,#e8590c); }
.meter .thr { position: absolute; top: -4px; bottom: -4px; width: 2px; background: #333; }
.meter .lab { display: flex; justify-content: space-between; font-size: .78rem; color: #5c6672; margin-top: 4px; }
.note { color: #5c6672; font-size: .84rem; margin-top: 10px; }
.mtable td.num { font-weight: 600; }
.tag { font-size: .72rem; font-weight: 600; padding: 2px 8px; border-radius: 6px; margin-left: 8px; vertical-align: middle; }
.tag.gt { background: #e7f7ec; color: #1b7a3d; }
.tag.lf { background: #e5efff; color: #1f57bd; }
.tag.cons { background: #fff4e5; color: #8a4b00; }
"""

def nav(active: str) -> str:
    a = ' class="active"' if active == "analyse" else ""
    b = ' class="active"' if active == "results" else ""
    return (f'<nav><a href="/"{a}>Analyse recording</a>'
            f'<a href="/results"{b}>Model metrics</a></nav>')

INDEX_HTML = """
<!doctype html><html><head><meta charset="utf-8">
<title>Pran · ICP-modulation demo</title><style>{{ css }}</style></head>
<body><div class="wrap">
  {{ nav|safe }}
  <h1>Pran - ICP-modulation demo</h1>
  <p class="sub">Zero-shot CHARIS clinical model (trained on invasive ICP ground truth) applied to an optical TM recording.</p>
  <div class="banner"><b>Scope:</b> this reports a <b>relative</b> ICP-modulation trend across the
  provocation manoeuvres. It is <b>not</b> a diagnosis and <b>not</b> a calibrated mmHg value.</div>
  <div class="card">
    <form method="post" action="/analyze" enctype="multipart/form-data">
      <label class="drop">
        Upload a hardware recording (.csv with <code>ir_raw, disp_raw, artifact_flag, session_label</code>)
        <br><input type="file" name="file" accept=".csv" required>
      </label>
      <button type="submit">Analyse recording</button>
    </form>
  </div>
  {% if examples %}
  <div class="card">
    <b>No file handy?</b> Try one of the bundled recordings:
    <table>
      {% for e in examples %}
      <tr><td>{{ e }}</td>
      <td class="num"><form method="post" action="/analyze">
        <input type="hidden" name="example" value="{{ e }}">
        <button type="submit" style="padding:6px 14px;margin:0;font-size:.85rem;">Analyse</button>
      </form></td></tr>
      {% endfor %}
    </table>
  </div>
  {% endif %}
  <p class="foot">Model: CHARIS zero-shot (LOPO AUC 0.961 vs invasive ICP). No training runs here - the saved model is reused.</p>
</div></body></html>
"""

RESULT_HTML = """
<!doctype html><html><head><meta charset="utf-8">
<title>Result · {{ name }}</title><style>{{ css }}</style></head>
<body><div class="wrap">
  {{ nav|safe }}
  <h1 style="margin-top:6px;">Report - {{ name }}</h1>
  {% if not r.ok %}
    <div class="err">Could not analyse this file: {{ r.error }}</div>
  {% else %}
  <div class="banner"><b>Relative ICP-modulation proxy</b> - validated by physiology, not calibrated to mmHg. Not a diagnosis.</div>
  <div class="card">
    <div class="flagbox">
      <span class="badge {{ 'flag' if r.screening_flag else 'normal' }}">
        {{ 'FLAGGED - elevated' if r.screening_flag else 'NORMAL' }}
      </span>
      <div class="meter">
        <div class="bar">
          <div class="fill" style="width: {{ (100*r.screening_prob)|round(1) }}%;"></div>
          <div class="thr" style="left: {{ (100*r.screening_threshold)|round(1) }}%;"></div>
        </div>
        <div class="lab"><span>P(ICP elevated) = <b>{{ '%.3f'|format(r.screening_prob) }}</b></span>
             <span>screening threshold {{ '%.3f'|format(r.screening_threshold) }}</span></div>
      </div>
    </div>
    <div class="note">Screening flag from the CHARIS threshold on mean P(ICP elevated).
      <b>Not a diagnosis</b> - no invasive ground truth exists for this sensor, and the between-subject
      flag is age-confounded. The trustworthy signal is the <b>within-subject</b> dose-response below.</div>
  </div>
  <div class="card">
    <div class="verdict {{ 'ok' if r.tracks else 'weak' }}">{{ r.verdict }}</div>
    <div class="kv" style="margin-top:16px;">
      <div><span class="k">Windows analysed</span><span class="v">{{ r.n_windows }}</span></div>
      <div><span class="k">Within-subject ρ</span><span class="v">{{ '%+.2f'|format(r.rho) if r.rho is not none else '-' }}</span></div>
      <div><span class="k">Monotonic ladder</span><span class="v">{{ 'yes' if r.monotonic else 'no' }}</span></div>
      {% if r.valsalva_higher is not none %}
      <div><span class="k">Valsalva &gt; baseline</span><span class="v">{{ 'yes' if r.valsalva_higher else 'no' }}</span></div>
      {% endif %}
    </div>
  </div>
  <div class="card">
    <img src="data:image/png;base64,{{ plot }}" alt="ICP modulation plot">
  </div>
  <div class="card">
    <table>
      <tr><th>Manoeuvre (ascending expected ICP)</th><th class="num">ICP-elevation score</th><th class="num">slow-wave power</th></tr>
      {% for s in r.sessions_present %}
      <tr><td>{{ names[s] }}</td>
          <td class="num">{{ '%.3f'|format(r.per_session[s]) }}</td>
          <td class="num">{{ '%.3f'|format(r.slow_wave[s]) }}</td></tr>
      {% endfor %}
    </table>
  </div>
  {% endif %}
  <p class="foot">Model: CHARIS zero-shot · saved model reused, no training.</p>
</div></body></html>
"""


# ── model metrics (all verified against results/*.json this build) ───────────
MODEL_METRICS = [
    {
        "title": "Clinical ICP model - CHARIS",
        "tag": ("gt", "invasive ground truth"),
        "subtitle": "XGBoost trained on the invasive CHARIS ICP database. The one model with real pressure-bolt ground truth - fully non-circular.",
        "rows": [
            ("LOPO AUC (13 patients)", "0.961", "95% CI 0.924-0.985"),
            ("Held-out test AUC", "0.979", ""),
            ("F1 (test)", "0.804", ""),
            ("Recall / Specificity", "0.876 / 0.956", ""),
            ("Model output vs invasive ICP (Spearman ρ)", "0.94", "p ≈ 0"),
        ],
    },
    {
        "title": "Device - within-subject ICP tracking",
        "tag": ("lf"),
        "subtitle": "The clinical model applied blind to the optical sensor (never trained on hardware). Each subject is their own control, so age/HR confounds are removed by design.",
        "rows": [
            ("Within-subject ρ (3-level ladder)", "+0.95", "89% strictly monotonic"),
            ("Within-subject ρ (4-level ladder)", "+0.84", ""),
            ("Valsalva > baseline", "146 / 146", "p ≈ 0"),
            ("Friedman omnibus", "χ² = 339", "p < 10⁻¹⁶"),
            ("Postural-only ρ (drift-controlled)", "+0.61", "biomarker anti-drift p < 10⁻⁴"),
        ],
    },
    {
        "title": "Device - screening classifier (balanced)",
        "tag": ("cons"),
        "subtitle": "50:50 real-data balanced model for the NORMAL/FLAGGED screening call. Hardware labels are model-derived, so these are screening/consistency numbers - not diagnostic accuracy.",
        "rows": [
            ("LOPO AUC", "0.904", "vs model-derived labels"),
            ("Average precision", "0.390", ""),
            ("F1 / Precision / Recall", "0.435 / 0.412 / 0.461", "positives ~5% prevalent"),
            ("Matthews corr. coef. (MCC)", "0.405", ""),
            ("Within-subject dose ρ (3-level)", "+0.85", "Valsalva 100%"),
        ],
    },
]

RESULTS_HTML = """
<!doctype html><html><head><meta charset="utf-8">
<title>Pran · model metrics</title><style>{{ css }}</style></head>
<body><div class="wrap">
  {{ nav|safe }}
  <h1 style="margin-top:6px;">Model metrics</h1>
  <p class="sub">Validated, leakage-free numbers. Tags mark what each metric is measured against.</p>
  <div class="banner"><b>How to read this:</b>
    <span class="tag gt">invasive ground truth</span> = checked against real ICP;
    <span class="tag lf">label-free</span> = physiology-validated, no labels;
    <span class="tag cons">screening</span> = device has no invasive ground truth, so it is a screening/consistency metric, not diagnostic accuracy.</div>
  {% for m in models %}
  <div class="card">
    <h2 style="margin:0 0 4px; font-size:1.15rem;">{{ m.title }}
      <span class="tag {{ m.tag[0] }}">{{ m.tag[1] }}</span></h2>
    <p class="note" style="margin-top:2px;">{{ m.subtitle }}</p>
    <table class="mtable">
      {% for label, val, extra in m.rows %}
      <tr><td>{{ label }}</td><td class="num">{{ val }}</td>
          <td style="color:#8a929c;font-size:.85rem;">{{ extra }}</td></tr>
      {% endfor %}
    </table>
  </div>
  {% endfor %}
  <p class="foot">Sources: results/qt_pipeline, results/invasive_validation, results/two_model_comparison, results/hybrid_balanced. Broken/uninformative diagnostics (e.g. inverted-calibration baselines) are omitted; nothing is relabelled.</p>
</div></body></html>
"""


@app.route("/")
def index():
    examples = sorted(p.name for p in Path("hw-tests").glob("*.csv"))[:8] \
        if Path("hw-tests").exists() else []
    return render_template_string(INDEX_HTML, css=BASE_CSS, nav=nav("analyse"), examples=examples)


@app.route("/analyze", methods=["POST"])
def analyze():
    # Either an uploaded file or a bundled example
    example = request.form.get("example")
    if example:
        path = Path("hw-tests") / Path(example).name
        name = path.stem
        if not path.exists():
            return render_template_string(RESULT_HTML, css=BASE_CSS, nav=nav("analyse"), name=name,
                                          names=SESSION_NAME, r={"ok": False, "error": "example not found"})
    else:
        f = request.files.get("file")
        if not f or not f.filename.lower().endswith(".csv"):
            return render_template_string(RESULT_HTML, css=BASE_CSS, nav=nav("analyse"), name="(no file)",
                                          names=SESSION_NAME, r={"ok": False, "error": "please choose a .csv file"})
        path = UPLOAD_DIR / Path(f.filename).name
        f.save(path)
        name = path.stem

    r = predict.analyse_csv(path)
    plot = make_plot(r) if r["ok"] else None
    return render_template_string(RESULT_HTML, css=BASE_CSS, nav=nav("analyse"), name=name,
                                  names=SESSION_NAME, r=r, plot=plot)


@app.route("/results")
def results():
    return render_template_string(RESULTS_HTML, css=BASE_CSS, nav=nav("results"), models=MODEL_METRICS)


def _open_browser():
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    print("Loading saved model (no training) ...")
    predict.load_model()  # warm the cache once, up front
    print("Pran demo running →  http://127.0.0.1:5000   (Ctrl+C to stop)")
    threading.Timer(1.0, _open_browser).start()
    app.run(host="127.0.0.1", port=5000, debug=False)
