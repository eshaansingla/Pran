#!/usr/bin/env bash
# =============================================================================
# run_full_pipeline.sh
# Complete ICP monitoring pipeline: data → train → evaluate → hardware demo
#
# Usage:
#   bash run_full_pipeline.sh            # auto-detects credentials/synthetic
#   bash run_full_pipeline.sh --synthetic # force synthetic data
#   bash run_full_pipeline.sh --download-only
# =============================================================================
set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()      { echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
section() { echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; \
            echo -e "${GREEN}  $*${NC}"; \
            echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }

# ── Argument parsing ──────────────────────────────────────────────────────────
SYNTHETIC=0
DOWNLOAD_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --synthetic)      SYNTHETIC=1 ;;
    --download-only)  DOWNLOAD_ONLY=1 ;;
  esac
done

# ── Auto-detect credentials ───────────────────────────────────────────────────
if [[ "$SYNTHETIC" -eq 0 ]]; then
  if [[ -z "${PHYSIONET_USERNAME:-}" ]] || [[ -z "${PHYSIONET_PASSWORD:-}" ]]; then
    warn "PHYSIONET_USERNAME / PHYSIONET_PASSWORD not set."
    warn "Falling back to synthetic data mode."
    warn "To use real data: export PHYSIONET_USERNAME=... PHYSIONET_PASSWORD=..."
    SYNTHETIC=1
  else
    info "PhysioNet credentials found for user: ${PHYSIONET_USERNAME}"
  fi
fi

# ── Create directory structure ────────────────────────────────────────────────
section "Setting up directories"
mkdir -p data/raw/charis data/raw/mimic data/processed \
         models/xgboost firmware/esp32_icp_monitor \
         results logs
ok "Directories ready"

# ── PHASE 1: Data ─────────────────────────────────────────────────────────────
section "Phase 1: Preparing training data"

if [[ "$SYNTHETIC" -eq 1 ]]; then
  info "Generating synthetic correlated dataset (50 patients × 200 windows) …"
  python src/data/save_processed_data.py \
    --synthetic \
    --n_patients 50 \
    --windows_each 200 \
    --output_dir data/processed \
    --log_dir logs
  ok "Synthetic dataset ready in data/processed/"
else
  info "Downloading CHARIS from PhysioNet …"
  python src/data/download_physionet.py
  ok "Download complete"

  if [[ "$DOWNLOAD_ONLY" -eq 1 ]]; then
    ok "Download-only mode — exiting."
    exit 0
  fi

  info "Segmenting & extracting features from raw records …"
  python src/data/save_processed_data.py \
    --charis_dir data/raw/charis \
    --mimic_dir  data/raw/mimic \
    --output_dir data/processed \
    --log_dir    logs
  ok "Feature extraction complete"
fi

# ── PHASE 2: Training ─────────────────────────────────────────────────────────
section "Phase 2: Training XGBoost classifier"

info "Training model (50 trees, max_depth=4, early stopping) …"
python src/models/xgboost_classifier.py \
  --mode       train \
  --data_dir   data/processed \
  --output_dir models/xgboost \
  --results_dir results \
  --log_dir    logs
ok "Model trained → models/xgboost/xgboost_best.pkl"

# ── Ablation study ────────────────────────────────────────────────────────────
section "Phase 2b: Ablation study (phase-lag feature contribution)"
python src/models/xgboost_classifier.py \
  --mode        ablation \
  --data_dir    data/processed \
  --results_dir results \
  --log_dir     logs
ok "Ablation complete → results/ablation_study.json"

# ── SHAP ──────────────────────────────────────────────────────────────────────
section "Phase 2c: SHAP feature importance"
python src/models/xgboost_classifier.py \
  --mode        shap \
  --data_dir    data/processed \
  --model_path  models/xgboost/xgboost_best.pkl \
  --results_dir results \
  --log_dir     logs
ok "SHAP plot → results/shap_feature_importance.png"

# ── C export ──────────────────────────────────────────────────────────────────
section "Phase 2d: Export model to C (ESP32 deployment)"
python src/models/xgboost_classifier.py \
  --mode       export_c \
  --data_dir   data/processed \
  --model_path models/xgboost/xgboost_best.pkl \
  --c_output   firmware/esp32_icp_monitor/xgboost_model.h \
  --log_dir    logs
ok "C header → firmware/esp32_icp_monitor/xgboost_model.h"

# ── PHASE 3: Hardware prediction demo ────────────────────────────────────────
section "Phase 3: Hardware prediction demo"

if [[ ! -f "data/sample_hardware_data.csv" ]]; then
  warn "data/sample_hardware_data.csv not found — generating sample …"
  python - <<'PYEOF'
import pandas as pd, numpy as np, pathlib
# 5 sample windows: 2 Normal, 1 Elevated, 2 Critical
rows = [
    [34.2, 1.18, 7.9, 0.11, 0.34, 81.0,  0, 0, -0.31, 0.48, 0.77],
    [31.5, 1.22, 8.3, 0.13, 0.37, 84.0,  0, 0, -0.28, 0.51, 0.74],
    [21.3, 1.09, 5.8, 0.41, 0.49, 98.0,  0, 0, -0.09, 0.69, 0.61],
    [ 7.8, 1.01, 3.9, 0.73, 0.64, 112.0, 0, 0,  0.22, 0.91, 0.44],
    [ 5.2, 0.98, 3.2, 0.79, 0.67, 118.0, 0, 0,  0.35, 0.94, 0.40],
]
cols = ["cardiac_amplitude","cardiac_frequency","respiratory_amplitude",
        "slow_wave_power","cardiac_power","mean_arterial_pressure",
        "head_angle","motion_artifact_flag","phase_lag_mean",
        "phase_lag_std","phase_coherence"]
pathlib.Path("data").mkdir(exist_ok=True)
pd.DataFrame(rows, columns=cols).to_csv("data/sample_hardware_data.csv", index=False)
print("  Sample hardware data created.")
PYEOF
fi

info "Running hardware predictions on sample data …"
python predict_from_hardware.py \
  --input data/sample_hardware_data.csv \
  --model models/xgboost/xgboost_best.pkl

# ── Summary ───────────────────────────────────────────────────────────────────
section "Pipeline Complete!"

echo ""
echo "  Output files:"
echo ""
printf "  %-42s" "models/xgboost/xgboost_best.pkl"
[[ -f models/xgboost/xgboost_best.pkl ]] && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}"

printf "  %-42s" "firmware/esp32_icp_monitor/xgboost_model.h"
[[ -f firmware/esp32_icp_monitor/xgboost_model.h ]] && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}"

for f in results/classification_report.txt results/ablation_study.json \
          results/shap_feature_importance.png; do
  printf "  %-42s" "$f"
  [[ -f "$f" ]] && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}"
done

echo ""
echo "  To predict from your own hardware data:"
echo "    python predict_from_hardware.py --input your_sensor_data.csv"
echo ""
