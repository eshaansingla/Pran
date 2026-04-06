$Root = (Resolve-Path "$PSScriptRoot\..\..")

$env:MODEL_PATH = "$Root\models\xgboost_binary.pkl.gz"
Write-Host "Backend starting on http://localhost:8001" -ForegroundColor Green
python -m uvicorn main:app --reload --port 8001
