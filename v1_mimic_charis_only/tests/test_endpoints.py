"""Integration tests for FastAPI endpoints."""
import sys
from pathlib import Path

import pytest

# Skip all tests if model not available
MODEL_PATH = Path(__file__).parent.parent / "models" / "xgboost_binary.pkl.gz"

try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "icp-monitor-web" / "backend"))
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    HAS_APP = True
except Exception:
    HAS_APP = False
    client = None

pytestmark = pytest.mark.skipif(
    not HAS_APP or not MODEL_PATH.exists(),
    reason="FastAPI app or model not available"
)


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert "timestamp" in r.json()


class TestModelInfo:
    def test_model_info_returns_data(self):
        r = client.get("/api/model_info")
        assert r.status_code == 200
        data = r.json()
        assert "version" in data
        assert "features" in data
        assert len(data["features"]) == 6


class TestPredictEndpoint:
    def test_valid_prediction(self, sample_features):
        r = client.post("/api/predict", json={"features": sample_features})
        assert r.status_code == 200
        data = r.json()
        assert data["class"] in (0, 1)
        assert 0 <= data["probability"] <= 1

    def test_wrong_feature_count(self):
        r = client.post("/api/predict", json={"features": [1.0, 2.0]})
        assert r.status_code == 422

    def test_missing_features(self):
        r = client.post("/api/predict", json={})
        assert r.status_code == 422


class TestPredictBatch:
    def test_valid_csv_upload(self, sample_csv_bytes):
        r = client.post(
            "/api/predict_batch",
            files={"file": ("test.csv", sample_csv_bytes, "text/csv")},
        )
        assert r.status_code == 200
        data = r.json()
        assert "predictions" in data
        assert "summary" in data
        assert data["summary"]["total"] > 0

    def test_non_csv_rejected(self):
        r = client.post(
            "/api/predict_batch",
            files={"file": ("test.txt", b"not csv", "text/plain")},
        )
        assert r.status_code == 415

    def test_empty_csv(self):
        r = client.post(
            "/api/predict_batch",
            files={"file": ("test.csv", b"", "text/csv")},
        )
        assert r.status_code == 422


class TestForecastEndpoint:
    def test_insufficient_windows(self):
        r = client.post("/api/predict_forecast",
                        json={"sequence": [[1.0]*6]*5})
        assert r.status_code == 422  # validation error: need >= 30

    def test_wrong_feature_count_in_sequence(self):
        seq = [[1.0, 2.0, 3.0]] * 30  # 3 features instead of 6
        r = client.post("/api/predict_forecast", json={"sequence": seq})
        assert r.status_code == 422


class TestExampleCsv:
    def test_example_csv_returns_data(self):
        r = client.get("/api/example_csv")
        assert r.status_code == 200
        data = r.json()
        assert "csv" in data
        assert "header" in data
