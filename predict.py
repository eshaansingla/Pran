"""predict.py - analyse one hardware recording with the trained hardware model.

    python predict.py hw-tests/icp_100_20_M.csv      one recording, detailed report
    python predict.py --all                           every recording in hw-tests/ (in-sample summary)

The model recognises the recorded *state* (Valsalva / the higher-ICP half of the protocol) from optical
features. It is not an ICP measurement, not calibrated to mmHg, and not a diagnosis.
Held-out performance numbers are in models/hardware/meta.json (also shown on the app's Results page).
"""
from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from pran.features import ICP_ORDER, REQUIRED_COLUMNS, SESSION_NAMES, recording_windows
from pran.hardware_model import HardwareModel, trailing_mean

WINDOW_SECONDS, STEP_SECONDS = 10, 5


@lru_cache(maxsize=1)
def load_model() -> HardwareModel:
    return HardwareModel()


def _subject_info(name: str) -> dict:
    m = re.match(r"icp_(\d+)_(\d+)_([MF])", name, re.IGNORECASE)
    return {"subject": int(m.group(1)), "age": int(m.group(2)), "sex": m.group(3).upper()} if m else {}


def analyse_csv(path: str | Path) -> dict:
    """Score one recording. Returns a dict for the CLI and the web app (all arrays are plain lists)."""
    path = Path(path)
    df = pd.read_csv(path, comment="#", low_memory=False)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        return {"ok": False, "error": "missing required columns: " + ", ".join(sorted(missing))}
    X, sess = recording_windows(df)
    if len(X) == 0:
        return {"ok": False, "error": "no usable signal windows in this recording (flat or artefact-flagged)"}
    model = load_model()
    v, lad = model.score(X)
    v35 = trailing_mean(v)
    flagged = v >= model.threshold
    out = {"ok": True, "name": path.name, "info": _subject_info(path.name), "n_windows": int(len(X)),
           "minutes": round(len(X) * STEP_SECONDS / 60 + 5 / 60, 1), "threshold": model.threshold,
           "in_sample": (Path("hw-tests") / path.name).exists(),
           "valsalva": v.tolist(), "valsalva_35s": v35.tolist(), "ladder": lad.tolist(),
           "pct_flagged_overall": round(float(flagged.mean() * 100), 1), "sessions": None}
    if sess is None:
        return out
    per, present = {}, [s for s in ICP_ORDER if (sess == s).sum() > 0]
    for s in present:
        m = sess == s
        per[s] = {"name": SESSION_NAMES[s], "n": int(m.sum()), "pct_flagged": round(float(flagged[m].mean() * 100), 1),
                  "mean_valsalva": round(float(v[m].mean()), 3), "mean_ladder": round(float(lad[m].mean()), 3)}
    out["sessions"] = sess.tolist(); out["per_session"] = per; out["present"] = present
    if len(present) >= 2:
        top = max(present, key=lambda s: per[s]["mean_valsalva"]); out["highest_session"] = top
        out["valsalva_is_highest"] = (top == 3) if 3 in present else None
    if len(present) == 4:
        ladder_means = [per[s]["mean_ladder"] for s in ICP_ORDER]
        out["ladder_monotone"] = all(ladder_means[i] < ladder_means[i + 1] for i in range(3))
    return out


def print_report(r: dict) -> None:
    if not r["ok"]:
        print("  [skip]", r["error"]); return
    line = "=" * 64
    print(line); print(f"  HARDWARE RECORDING REPORT - {r['name']}"); print(line)
    if r["info"]: print(f"  subject {r['info']['subject']} | age {r['info']['age']} | sex {r['info']['sex']}")
    print(f"  windows analysed : {r['n_windows']} (~{r['minutes']} min)   window threshold {r['threshold']:.3f}")
    if r.get("per_session"):
        print(f"\n  {'session (ICP order)':<20}{'windows':>8}{'% flagged':>11}{'Valsalva score':>16}{'ladder score':>14}"); print("  " + "-" * 67)
        for s in r["present"]:
            p = r["per_session"][s]; print(f"  {p['name']:<20}{p['n']:>8}{p['pct_flagged']:>10.1f}%{p['mean_valsalva']:>16.3f}{p['mean_ladder']:>14.3f}")
        if r.get("valsalva_is_highest") is not None:
            print(f"\n  Highest Valsalva score : {SESSION_NAMES[r['highest_session']]}  ->  {'as expected' if r['valsalva_is_highest'] else 'NOT the Valsalva session'}")
        if "ladder_monotone" in r:
            print(f"  Ladder head-up < supine < head-down < Valsalva : {'yes' if r['ladder_monotone'] else 'no'}")
    else:
        print(f"  windows flagged Valsalva-like: {r['pct_flagged_overall']}%  (no session labels in this file)")
    if r["in_sample"]: print("\n  NOTE: this recording was part of the training set (in-sample). Held-out results: models/hardware/meta.json")
    print("  Recognises the recorded state; not an ICP measurement, not calibrated to mmHg, not a diagnosis.")


def main(argv: list[str]) -> None:
    if not argv:
        print(__doc__); return
    if argv[0] == "--all":
        files = sorted(Path("hw-tests").glob("icp_*.csv")); res = [analyse_csv(f) for f in files]; ok = [r for r in res if r["ok"] and r.get("valsalva_is_highest") is not None]
        print(f"Analysed {len(ok)} recordings (in-sample): Valsalva session highest in {sum(r['valsalva_is_highest'] for r in ok)}/{len(ok)}; "
              f"ladder monotone in {sum(r.get('ladder_monotone', False) for r in ok)}/{len(ok)}.  Held-out numbers: models/hardware/meta.json")
    else:
        print_report(analyse_csv(argv[0]))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main(sys.argv[1:])
