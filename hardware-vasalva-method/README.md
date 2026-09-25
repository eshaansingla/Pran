# hardware-vasalva-method

Hardware recordings (146 volunteers, 46,407 ten-second windows, 32 optical features). **Valsalva windows = abnormal (1), all other windows (supine, head-up, head-down) = normal (0).** 14 models, same protocol, one folder per model.

> **What this is and is not.** The label is the *recorded manoeuvre*, not a measured ICP. The models recognise the Valsalva state from optical features. This is not an ICP measurement, not calibrated to mmHg, and not a diagnosis. The signal also contains systemic blood-flow effects (IR and red channels alone each reach ~0.98).

## Protocol (leakage controls)
1. **Final test set:** 29 subjects (age-stratified, seed 42) set aside first and never used for any choice. Each model is scored on them **once**.
2. **Development:** the other 117 subjects, 10-fold **subject-grouped** CV. Per fold, three disjoint subject sets: fit, 14 validation subjects (early stopping + Youden threshold), held-out (asserted in code).
3. **Scaler** (QuantileTransformer) fit on the fitting subjects only. Hyper-parameters fixed in advance, no tuning on test data. No sklearn built-in early stopping (it splits by window, which would leak inside a subject).
4. **Winner** = highest pooled out-of-fold AUC on development subjects (rule fixed before running). Confirmed on the untouched final test.
5. Features are per-window (detrended and normalised within the window). The session label is never a feature.

## Results (AUC; sensitivity/specificity at the validation-chosen threshold)
| Model | Dev AUC (95% CI over subjects) | Final-test AUC | Dev sens / spec | Test sens / spec | Train-vs-held-out gap |
|---|---|---|---|---|---|
| **XGBoost** | **0.994** (0.992-0.996) | **0.995** (0.989-0.998) | 0.970 / 0.949 | 0.957 / 0.967 | +0.006 |
| LightGBM | 0.994 | 0.995 | 0.970 / 0.948 | 0.958 / 0.971 | +0.006 |
| HistGradBoost | 0.994 | 0.995 | 0.967 / 0.953 | 0.955 / 0.965 | +0.006 |
| CatBoost | 0.993 | 0.995 | 0.968 / 0.947 | 0.957 / 0.970 | +0.006 |
| GradientBoosting | 0.991 | 0.992 | 0.963 / 0.937 | 0.931 / 0.964 | +0.005 |
| MLP (64,32) | 0.989 | 0.992 | 0.951 / 0.949 | 0.954 / 0.956 | +0.010 |
| ExtraTrees / RandomForest | 0.988 / 0.988 | 0.989 / 0.989 | 0.955 / 0.935, 0.958 / 0.925 | | +0.008 / +0.010 |
| AdaBoost / KNN | 0.985 / 0.984 | 0.987 / 0.984 | | | |
| LinearSVM / LogisticRegression | 0.982 / 0.982 | 0.981 / 0.982 | | | |
| DecisionTree | 0.968 | 0.972 | | | +0.017 |
| NaiveBayes | 0.955 | 0.960 | | | 0.000 |

Full table: `summary.csv`; figures: `ranking_auc.png`, `metrics_heatmap.png`, `overfit_underfit_check.png`, `sensitivity_vs_specificity.png`, `end_results_table.png`.

**Best model: XGBoost, but XGBoost, LightGBM, HistGradBoost and CatBoost are within 0.001 AUC of each other** (overlapping CIs; LightGBM is statistically indistinguishable, Holm p = 1.0). Say "top group", not "XGBoost beats the rest". Simple models (logistic regression, linear SVM, 0.982) are close behind, which shows the effect is strong and mostly linear-separable; a single tree (0.968) and Naive Bayes (0.955) are clearly worse.

## Leakage / overfitting / underfitting audit
| Check | Result |
|---|---|
| Development vs untouched final test | 0.994 vs 0.995 (no selection leakage; test not worse) |
| Label-shuffle null (labels shuffled inside each subject) | AUC 0.501 (a leaking pipeline would stay high) |
| Train vs held-out AUC | gap +0.006 for the winner; no model exceeds 0.017 (overfit rule: gap > 0.05) |
| Underfitting | none: learning curve 0.960 (8 subjects) -> 0.989 (32) -> 0.994 (64) -> 0.994 (92): flat at the top |
| Held-out **blocks of consecutive subject IDs** (recording batches) | AUC 0.989 / 0.990 / 0.996 / 0.989 / 0.998 |
| Train on early IDs, test on late (and reverse) | 0.992 / 0.990 |
| Duplicate recordings across files (shared raw sample runs) | none found |
| Which signals carry it | 5 CHARIS-style features alone 0.703; IR only 0.979; displacement only 0.959; red only 0.976; the 27 optical features without the 5 base ones 0.994 |
| Early vs late *inside the supine session* (model score) | AUC 0.503: the model does not track elapsed time |
| Valsalva vs each other session | supine 0.998, head-up 0.9996, head-down 0.981 (development; 0.999 / 1.000 / 0.984 on final test) |
| By age | <=18: 0.959, 19-25: 0.999, 26-45: 0.996, 46-64: 0.997, 65+: 0.973; sex: F 0.989, M 0.996 |

Files: `controls_best.json/.png`, `leak_hunt.json`.

## Honest caveats (what no split can fix)
- **Fixed manoeuvre order.** Valsalva is always the last block: position-in-recording alone scores AUC **1.000**. Any drift (warm-up, fatigue, sensor settling) is perfectly aligned with the label. The within-supine drift check (0.503) and the batch holdouts argue against a drift artefact, but only a randomised-order study would settle it.
- **A large, easy perturbation.** Valsalva changes venous return, blood pressure and pulse strongly, so 0.99 is plausible for *state recognition*. It says nothing about intracranial pressure specifically.
- **Boundary windows** that straddle two sessions are labelled by majority (a little label noise). Single site, single device, young cohort (median age 21); 65+ and <=18 groups are smaller and score a little lower.
- The single-tree and naive-Bayes rows show the models are not just memorising: weaker models score lower on both development and test.

## Files
`common.py` (data, splits, models, metrics), `run_all.py` (train/evaluate all 14), `plots.py`, `controls_best.py`, `leak_hunt.py`. Per model: `metrics.json`, `per_fold.csv`, `per_subject.csv`, `predictions.csv` (development out-of-fold and final-test scores), `results.png`, `feature_importance.png`, `model.pkl`, `qt_scaler.pkl` (trained on development subjects; its evaluation is the table above). `_partial/` is a resume cache and can be deleted.

Run from the repo root: `python hardware-vasalva-method/run_all.py`, then `plots.py`, `controls_best.py`, `leak_hunt.py`.
