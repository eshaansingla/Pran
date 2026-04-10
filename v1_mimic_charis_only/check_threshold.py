import json
m = json.load(open("models/binary_meta.json"))
print(f"Threshold: {m['prob_threshold']}")
print(f"Recall: {m['metrics']['recall']:.4f}")
print(f"Precision: {m['metrics']['precision']:.4f}")
print(f"F1: {m['metrics']['f1']:.4f}")
print(f"AUC: {m['metrics']['auc']:.4f}")
print(f"ECE before: {m['ece_before_calibration']}")
print(f"ECE after: {m['ece_after_calibration']}")
