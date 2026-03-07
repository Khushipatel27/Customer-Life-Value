"""
Unit tests for src/predict.py

Tests cover simulate_prediction and assign_rfm_segment without requiring
any saved model files on disk.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from predict import FEATURE_COLS, assign_rfm_segment, simulate_prediction
from train import build_pipeline

_RNG = np.random.default_rng(42)
_N = 30


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_features() -> pd.DataFrame:
    """Synthetic customer feature DataFrame — no file I/O required."""
    data = {
        "recency": _RNG.integers(1, 365, _N).astype(float),
        "frequency": _RNG.integers(2, 50, _N).astype(float),
        "monetary_mean": _RNG.uniform(10, 500, _N),
        "monetary_total": _RNG.uniform(50, 5000, _N),
        "avg_days_between_orders": _RNG.uniform(1, 90, _N),
        "num_unique_products": _RNG.integers(1, 30, _N).astype(float),
        "num_unique_countries": _RNG.integers(1, 3, _N).astype(float),
        "weekend_purchase_ratio": _RNG.uniform(0, 1, _N),
        "return_rate": _RNG.uniform(0, 0.3, _N),
        "first_purchase_recency": _RNG.integers(30, 730, _N).astype(float),
        "future_revenue": _RNG.uniform(0, 2000, _N),
    }
    return pd.DataFrame(
        data, index=pd.Index(range(10001, 10001 + _N), name="Customer ID")
    )


@pytest.fixture(scope="module")
def fitted_pipeline(small_features):
    pipeline = build_pipeline()
    X = small_features[FEATURE_COLS]
    y = small_features["future_revenue"]
    pipeline.fit(X, y)
    return pipeline


# ---------------------------------------------------------------------------
# simulate_prediction
# ---------------------------------------------------------------------------


def test_simulate_prediction_returns_float(small_features, fitted_pipeline):
    feat_dict = {col: float(small_features[col].iloc[0]) for col in FEATURE_COLS}
    result = simulate_prediction(feat_dict, fitted_pipeline)
    assert isinstance(result, float)


def test_simulate_prediction_is_finite(small_features, fitted_pipeline):
    feat_dict = {col: float(small_features[col].mean()) for col in FEATURE_COLS}
    result = simulate_prediction(feat_dict, fitted_pipeline)
    assert np.isfinite(result)


def test_simulate_prediction_changes_with_input(small_features, fitted_pipeline):
    """Two different feature vectors should (almost always) produce different CLVs."""
    base = {col: float(small_features[col].mean()) for col in FEATURE_COLS}
    high_monetary = {**base, "monetary_total": base["monetary_total"] * 10}
    pred_base = simulate_prediction(base, fitted_pipeline)
    pred_high = simulate_prediction(high_monetary, fitted_pipeline)
    assert pred_base != pred_high


# ---------------------------------------------------------------------------
# assign_rfm_segment
# ---------------------------------------------------------------------------


_VALID_SEGMENTS = {"Champions", "Loyal", "At Risk", "Promising", "Lost", "Others"}


def test_assign_rfm_segment_returns_valid_labels(small_features):
    segments = assign_rfm_segment(small_features)
    assert set(segments.unique()).issubset(_VALID_SEGMENTS)


def test_assign_rfm_segment_length_matches_input(small_features):
    segments = assign_rfm_segment(small_features)
    assert len(segments) == len(small_features)


def test_assign_rfm_segment_index_matches_input(small_features):
    segments = assign_rfm_segment(small_features)
    pd.testing.assert_index_equal(segments.index, small_features.index)


def test_assign_rfm_segment_no_nulls(small_features):
    segments = assign_rfm_segment(small_features)
    assert segments.isna().sum() == 0


def test_assign_rfm_segment_high_recency_high_freq_is_champion():
    """A customer with very low recency days and very high frequency should be a Champion."""
    df = pd.DataFrame(
        {
            "recency": [1.0] * 5 + [364.0] * 5,
            "frequency": [100.0] * 5 + [1.0] * 5,
            "monetary_mean": [500.0] * 10,
            "monetary_total": [5000.0] * 10,
            "avg_days_between_orders": [3.0] * 10,
            "num_unique_products": [50.0] * 10,
            "num_unique_countries": [1.0] * 10,
            "weekend_purchase_ratio": [0.2] * 10,
            "return_rate": [0.0] * 10,
            "first_purchase_recency": [400.0] * 10,
            "future_revenue": [1000.0] * 10,
        },
        index=pd.Index(range(1, 11), name="Customer ID"),
    )
    segments = assign_rfm_segment(df)
    # The first 5 rows (low recency days = recent, high frequency) should be Champions
    assert segments.iloc[0] == "Champions"
