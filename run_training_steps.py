"""
Run steps 2-8 of the overnight pipeline (extraction already done).
"""
import sys
sys.path.insert(0, '.')
from overnight_pipeline import (
    step2_clean_data, step3_retrain_xgboost, step4_baselines,
    step5_lstm_recompute, step6_firmware, step7_tests, step8_summary, log
)
from datetime import datetime

log(f"Starting post-extraction pipeline: {datetime.now().isoformat()}")
log(f"Extraction already done — 153 patients, 159086 windows")

results = {}
results["2_clean"]     = step2_clean_data()
results["3_xgboost"]   = step3_retrain_xgboost()
results["4_baselines"] = step4_baselines()
results["5_lstm"]      = step5_lstm_recompute()
results["6_firmware"]  = step6_firmware()
results["7_tests"]     = step7_tests()
results["8_summary"]   = step8_summary()

log("\n" + "=" * 70)
log("  STEP STATUS")
for step, ok in results.items():
    log(f"  {step:<20} {'✅ PASS' if ok else '❌ FAIL'}")
log(f"  Completed: {datetime.now().isoformat()}")
log("=" * 70)
