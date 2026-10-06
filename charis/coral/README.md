# CORAL-aligned CHARIS model

Domain-adapted version of `models/charis_compare/XGBoost`. Same 5 features, same
CHARIS labels, but CHARIS's training features are CORAL-aligned (whitened, then
re-colored with the hardware background's covariance) before the classifier is
fit, so the decision boundary depends less on CHARIS-only feature geometry.

**Trained only on CHARIS** (features + real ground-truth labels). The hardware
background (`support/results/hw_features_cache.npz`, 146 subjects, unlabeled) is
used only to define the covariance target for CORAL -- no hardware rows or
labels ever enter `clf.fit()`. See the training run in the parent conversation
for the full derivation.

## Files
- `train.py` -- refits everything from scratch, run from repo root
- `predict.py` -- scores hardware feature CSVs (`window_index/center_time_s/cardiac_amplitude/.../cardiac_power` columns)
- `model.pkl`, `qt_charis.pkl`, `qt_hw.pkl`, `coral.npz` -- fitted artifacts
- `metrics.json` -- threshold + validation summary and known caveats

## Usage
```
python charis/coral/predict.py hw_data/*.csv
python charis/coral/predict.py path/to/your.csv
```

## Known limitations (read before citing a number from this model)
- **Domain separability is reduced, not eliminated.** A classifier can still tell
  CHARIS windows from hardware windows at AUC ~0.82-0.87 even after this
  alignment. This is a structural ceiling: CHARIS's 5 features are all derived
  from one invasive ICP channel (so they're correlated with each other);
  hardware's 5 features come from two physically independent sensors (PPG +
  displacement), so they aren't. No further scaling closes this gap without
  fabricating data.
- **Patient 5 in CHARIS is unreliable.** LOPO AUC looks fine (0.745) until you
  remove that patient's own baseline level, at which point it drops to 0.464
  (below chance) -- meaning the "signal" there was mostly "which patient is
  this," not real elevation detection. Patient 4 is genuinely hard either way
  (~0.55-0.58), not a leak.
- **No hardware ground truth exists.** Every hardware score is a screening
  number, not a verified result. The only hardware check that has been
  independently validated end-to-end is within-subject Valsalva-vs-supine
  dose response (see `hardware-vasalva-method/`), because it never requires an
  absolute cross-domain judgment.
- **Postural physiology check failed on 2 of 3 axes.** Across the 146-subject
  hardware set: Valsalva scores correctly higher than supine (right
  direction, matches known physiology). Head-up scores *higher* than supine
  (should be lower -- sitting up should reduce ICP). Head-down is flat versus
  supine (should be higher). This points to the PPG-derived cardiac features
  likely picking up postural cardiovascular change rather than pure
  ICP-correlated signal.
