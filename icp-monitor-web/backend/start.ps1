$env:MODEL_PATH = "C:\Users\asus\Documents\GitHub\Pran\models\xgboost_binary.pkl.gz"
Write-Host "Backend starting on http://localhost:8001"
python -m uvicorn main:app --reload --port 8001
