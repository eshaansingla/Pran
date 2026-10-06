"""CHARIS leave-one-patient-out XGBoost, run fold by fold with readable terminal output.

    python charis/lopo_train_test.py

Same split / params / seeds as compare_models.py. Per fold: 1 patient is the TEST set (never seen), 2 patients are the
VALIDATION set (early stopping + Youden threshold), the remaining patients are TRAIN. Prints train / val / test metrics
per fold and a summary table, and writes results/charis_lopo/train_test.csv (used by lopo_windows_report.py).
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import QuantileTransformer
import compare_models as C

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "results" / "charis_lopo"; OUT.mkdir(parents=True, exist_ok=True)
X, y, pid = (np.load(ROOT / C.CACHE / f"{n}.npy") for n in ("X", "y", "pid"))
pats = sorted(np.unique(pid)); rows = []
BAR = "=" * 90


def stats(tag, yt, s, thr):
    tn, fp, fn, tp = confusion_matrix(yt, (s >= thr).astype(int), labels=[0, 1]).ravel()
    return {f"{tag}_n": len(yt), f"{tag}_auc": roc_auc_score(yt, s), f"{tag}_sens": tp / max(tp + fn, 1), f"{tag}_spec": tn / max(tn + fp, 1),
            f"{tag}_flagged": int(tp + fp), f"{tag}_not_flagged": int(tn + fn),
            f"{tag}_TP": int(tp), f"{tag}_FN": int(fn), f"{tag}_TN": int(tn), f"{tag}_FP": int(fp)}


def say(*a):
    print(*a, flush=True)


say(BAR)
say(f"CHARIS leave-one-patient-out | XGBoost | {len(pats)} folds | {len(y):,} windows, {y.mean():.1%} elevated (ICP >= 20 mmHg)")
say("Features: " + ", ".join(C.FEATURES))
say("Each fold: 1 patient TEST (never seen), 2 patients VALIDATION (early stopping + threshold), the rest TRAIN.")
say(BAR)

for i, p in enumerate(pats):
    t0 = time.time(); te = pid == p; others = [q for q in pats if q != p]
    vp = [others[(i * 2) % 12], others[(i * 2 + 1) % 12]]; vm = np.isin(pid, vp); fm = ~te & ~vm
    say(f"\n[FOLD {i + 1}/{len(pats)}]  test = patient {int(p)}")
    say(f"  train patients : {[int(q) for q in others if q not in vp]}")
    say(f"  val patients   : {[int(q) for q in vp]}")
    say(f"  windows        : train {fm.sum():>9,} ({y[fm].mean():5.1%} elevated) | val {vm.sum():>8,} ({y[vm].mean():5.1%}) | test {te.sum():>8,} ({y[te].mean():5.1%})")
    qt = QuantileTransformer(output_distribution="normal", random_state=C.SEED, n_quantiles=1000, subsample=200_000).fit(X[fm])
    Xf, Xv, Xt = (qt.transform(X[m]).astype(np.float32) for m in (fm, vm, te)); yf, yv, yt = y[fm], y[vm], y[te]
    say("  training XGBoost ...")
    m, it = C.fit_boost("XGBoost", Xf, yf, Xv, yv)
    curve = m.evals_result()["validation_0"]["auc"]
    pts = sorted({0, *range(49, len(curve), 50), it - 1, len(curve) - 1})
    say("  validation AUC by boosting round: " + "  ".join(f"r{k + 1}:{curve[k]:.3f}" for k in pts))
    say(f"  early stopping: best round {it} of {len(curve)} (stops when val AUC has not improved for 50 rounds)")
    thr = C.youden(yv, C.score(m, Xv))
    r = dict(patient=int(p), threshold=thr, iters=it, **stats("train", yf, C.score(m, Xf), thr),
             **stats("val", yv, C.score(m, Xv), thr), **stats("test", yt, C.score(m, Xt), thr))
    say(f"  decision threshold (Youden on val) = {thr:.3f}   (score >= threshold -> window FLAGGED as elevated)\n")
    say(f"  {'':6}{'windows':>10}{'truly normal':>14}{'truly elevated':>16}{'FLAGGED':>10}{'NOT flagged':>13}{'correct calls':>15}")
    for tag, nm in (("train", "TRAIN"), ("val", "VAL"), ("test", "TEST")):
        n = r[tag + "_n"]; ok = r[tag + "_TP"] + r[tag + "_TN"]
        say(f"  {nm:<6}{n:>10,}{r[tag + '_TN'] + r[tag + '_FP']:>14,}{r[tag + '_TP'] + r[tag + '_FN']:>16,}{r[tag + '_flagged']:>10,}{r[tag + '_not_flagged']:>13,}{ok / n:>15.1%}")
    say(f"\n  TEST patient {int(p)}:  of {r['test_TP'] + r['test_FN']:,} truly elevated windows -> {r['test_TP']:,} flagged, {r['test_FN']:,} missed"
        f"  |  of {r['test_TN'] + r['test_FP']:,} truly normal windows -> {r['test_TN']:,} not flagged, {r['test_FP']:,} falsely flagged"
        f"   [AUC {r['test_auc']:.3f}, {time.time() - t0:.0f}s]")
    rows.append(r); pd.DataFrame(rows).to_csv(OUT / "train_test.csv", index=False)

d = pd.DataFrame(rows)
say(f"\n{BAR}\nSUMMARY (test = held-out patient of each fold; every count is 10 s windows)\n{BAR}")
say(f"{'patient':>8}{'test windows':>14}{'truly elevated':>16}{'FLAGGED':>10}{'NOT flagged':>13}{'elevated caught':>17}{'normal cleared':>16}{'AUC':>7}")
for _, r in d.iterrows():
    say(f"{int(r.patient):>8}{int(r.test_n):>14,}{int(r.test_TP + r.test_FN):>16,}{int(r.test_flagged):>10,}{int(r.test_not_flagged):>13,}{r.test_sens:>17.1%}{r.test_spec:>16.1%}{r.test_auc:>7.3f}")
say(f"{'TOTAL':>8}{int(d.test_n.sum()):>14,}{int((d.test_TP + d.test_FN).sum()):>16,}{int(d.test_flagged.sum()):>10,}{int(d.test_not_flagged.sum()):>13,}{d.test_sens.mean():>17.1%}{d.test_spec.mean():>16.1%}{d.test_auc.mean():>7.3f}   (rates and AUC = mean over patients)")
say(f"Train AUC (mean) {d.train_auc.mean():.3f}  vs  test AUC (mean) {d.test_auc.mean():.3f}")
say(f"\nTest AUC >= 0.90 in {(d.test_auc >= .9).sum()}/{len(d)} patients; worst = patient {int(d.loc[d.test_auc.idxmin(), 'patient'])} ({d.test_auc.min():.3f}).")
say(f"Saved {OUT / 'train_test.csv'}  (run charis/lopo_windows_report.py to redraw the per-patient images)")
