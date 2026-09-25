# testing/ — cross-domain tests (CHARIS <-> hardware)

Nothing outside this folder was modified. Run from the repo root.

| File | What |
|---|---|
| `test_charis_on_hw.py` | Test A: CHARIS-trained XGBoost -> 146 hardware volunteers |
| `test_hw_on_charis.py` | Test B: hardware-trained XGBoost -> CHARIS patients |
| `A_*.json/.csv`, `B_*.json/.csv` | results (per subject / per patient) |
| `charis_hwstyle_features.npz` | cache of CHARIS features computed with the hardware extractor (every 4th window) |

## Test A — CHARIS model on hardware (Valsalva = 1, other sessions = 0)
Model: `models/charis_compare/XGBoost` (13 CHARIS patients, never saw hardware data), its own scaler and threshold (0.436).
- Pooled AUC **0.648** (Valsalva vs supine only: 0.650); mean within-subject AUC 0.659.
- Valsalva mean score above supine in 146/146 subjects, but the effect is small: windows flagged 9.6% in Valsalva vs 1.9% in supine; sensitivity 0.10, specificity 0.98.
- Age confound: 23.8% of windows flagged for age >65 vs 1.4% for age <30; the model separates >65 from <30 with AUC 0.81, i.e. it tracks age more strongly than the manoeuvre.

## Test B — hardware model on CHARIS (ICP >= 20 mmHg)
The saved hardware model needs 32 optical features (IR/displacement/red), which CHARIS does not have. So the hardware recipe was re-trained on the 5 base features only (hardware subjects only, same split/seed/hyper-parameters) and applied to CHARIS with hardware-style features (detrended, relative amplitudes; ICP waveform used as both channels). This is an approximation, not the saved 32-feature model.
- On its own held-out hardware subjects the 5-feature model gets AUC 0.712 (the ceiling for this feature set).
- On CHARIS: pooled AUC **0.408** (below chance), mean per-patient AUC **0.558**, median 0.590, worst 0.334; 9/13 patients above 0.5, 4 below. Sensitivity 0.20, specificity 0.80, balanced accuracy 0.50.

## Reading
- Neither direction transfers. Both land near chance to weakly-above-chance (0.65 for A, 0.56 per patient for B).
- The 0.994 hardware result depends on the 27 optical channel features; the 5 waveform features shared with CHARIS give only ~0.71 even inside hardware.
- Different tasks (Valsalva state vs ICP >= 20), different sensors and signals, and different populations. Chance-level transfer is expected and is not evidence that either result is wrong.
- Caveats: Test B uses subsampled CHARIS windows and a re-trained 5-feature model. The CHARIS model on disk is the one trained on raw-unit features (pre detrend fix), so Test A also reflects that feature mismatch. Single run, no repeated seeds.
