@echo off
set MODEL_PATH=%~dp0..\..\models\xgboost_final.pkl.gz
echo Starting ICP Monitor backend...
echo Model path: %MODEL_PATH%
uvicorn main:app --reload --port 8000
