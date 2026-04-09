import model_loader
m = model_loader.load_model()
print("Model loaded OK")
info = model_loader.get_model_info()
print(f"Total windows: {info['training_data']['total_windows']}")
print(f"AUC: {info['metrics']['auc']:.4f}")
print(f"F1: {info['metrics']['f1']:.4f}")
print(f"Calibrated: {info['calibrated']}")

# Test a sample prediction
sample = [25.0, 1.2, 8.0, 0.95, 0.01, 85.0]
result = model_loader.predict_single(sample)
print(f"Sample prediction: {result['class_name']} (p={result['probability']:.4f})")
