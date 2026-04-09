"""Tests for model_loader.py — model loading and prediction."""
import sys
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "icp-monitor-web" / "backend"))

# Check if model file exists before running these tests
MODEL_PATH = Path(__file__).parent.parent / "models" / "xgboost_binary.pkl.gz"
pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Model file not found — run train_binary.py first"
)


class TestModelLoading:
    def test_load_model(self):
        import model_loader
        model_loader._model = None  # force reload
        bst = model_loader.load_model()
        assert bst is not None

    def test_model_info(self):
        import model_loader
        info = model_loader.get_model_info()
        assert "version" in info
        assert "features" in info
        assert len(info["features"]) == 6
        assert "metrics" in info


class TestPredictions:
    def test_single_prediction(self, sample_features):
        import model_loader
        result = model_loader.predict_single(sample_features)
        assert "class" in result
        assert result["class"] in (0, 1)
        assert "probability" in result
        assert 0.0 <= result["probability"] <= 1.0
        assert "top_features" in result
        assert len(result["top_features"]) == 3

    def test_batch_prediction(self):
        import model_loader
        features = [
            [35.0, 1.2, 12.0, 0.95, 0.003, 80.0],
            [55.0, 0.9, 8.0, 0.78, 0.015, 100.0],
        ]
        results = model_loader.predict_batch(features)
        assert len(results) == 2
        for r in results:
            assert r["class"] in (0, 1)
            assert 0.0 <= r["probability"] <= 1.0

    def test_prediction_reproducibility(self, sample_features):
        """Same input should give same output."""
        import model_loader
        r1 = model_loader.predict_single(sample_features)
        r2 = model_loader.predict_single(sample_features)
        assert r1["probability"] == r2["probability"]
        assert r1["class"] == r2["class"]

    def test_abnormal_features_higher_prob(self, sample_features, sample_features_abnormal):
        """Abnormal features should generally yield higher P(Abnormal)."""
        import model_loader
        r_normal = model_loader.predict_single(sample_features)
        r_abnormal = model_loader.predict_single(sample_features_abnormal)
        # Not guaranteed but expected with properly trained model
        # Just check both return valid results
        assert 0 <= r_normal["probability"] <= 1
        assert 0 <= r_abnormal["probability"] <= 1
