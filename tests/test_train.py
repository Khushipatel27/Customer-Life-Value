"""
Unit tests for src/train.py

Tests cover metric computation and pipeline construction without loading
any external data files.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from train import build_pipeline, evaluate_metrics, mape


# ---------------------------------------------------------------------------
# mape
# ---------------------------------------------------------------------------


def test_mape_known_value():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])
    # |100-110|/100 = 10%,  |200-180|/200 = 10%  ->  mean = 10%
    assert mape(y_true, y_pred) == pytest.approx(10.0)


def test_mape_ignores_zero_actuals():
    y_true = np.array([0.0, 100.0])
    y_pred = np.array([50.0, 110.0])
    # Only the second element counts  ->  |100-110|/100 = 10%
    assert mape(y_true, y_pred) == pytest.approx(10.0)


def test_mape_all_zeros_returns_nan():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([1.0, 2.0])
    assert np.isnan(mape(y_true, y_pred))


def test_mape_perfect_prediction():
    y = np.array([50.0, 100.0, 200.0])
    assert mape(y, y) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# evaluate_metrics
# ---------------------------------------------------------------------------


def test_evaluate_metrics_perfect_prediction():
    y = np.array([100.0, 200.0, 300.0])
    m = evaluate_metrics(y, y)
    assert m["MAE"] == pytest.approx(0.0)
    assert m["RMSE"] == pytest.approx(0.0)
    assert m["R2"] == pytest.approx(1.0)
    assert m["Pearson_r"] == pytest.approx(1.0)


def test_evaluate_metrics_returns_all_keys():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9])
    m = evaluate_metrics(y_true, y_pred)
    assert set(m.keys()) == {"MAE", "RMSE", "R2", "MAPE", "Pearson_r"}


def test_evaluate_metrics_all_values_are_floats():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([11.0, 19.0, 31.0])
    m = evaluate_metrics(y_true, y_pred)
    for key, val in m.items():
        assert isinstance(val, float), f"{key} is not a float"


def test_evaluate_metrics_mae_is_positive():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 35.0])
    m = evaluate_metrics(y_true, y_pred)
    assert m["MAE"] > 0


def test_evaluate_metrics_rmse_gte_mae():
    """RMSE >= MAE always holds (RMSE penalises large errors more)."""
    y_true = np.array([10.0, 20.0, 100.0])
    y_pred = np.array([11.0, 25.0, 50.0])
    m = evaluate_metrics(y_true, y_pred)
    assert m["RMSE"] >= m["MAE"]


# ---------------------------------------------------------------------------
# build_pipeline
# ---------------------------------------------------------------------------


def test_build_pipeline_has_correct_steps():
    pipeline = build_pipeline()
    assert list(pipeline.named_steps.keys()) == ["scaler", "model"]


def test_build_pipeline_fits_and_predicts_on_tiny_data():
    pipeline = build_pipeline()
    X = pd.DataFrame(
        {col: np.random.rand(30) for col in
         ["recency", "frequency", "monetary_mean", "monetary_total",
          "avg_days_between_orders", "num_unique_products", "num_unique_countries",
          "weekend_purchase_ratio", "return_rate", "first_purchase_recency"]}
    )
    y = pd.Series(np.random.rand(30) * 1000)
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == 30
    assert not np.any(np.isnan(preds))
