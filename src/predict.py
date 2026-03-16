"""
Inference utilities for the CLV prediction model.

Used by the Streamlit app to load artefacts and run per-customer predictions.
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = ROOT_DIR / "outputs"

FEATURE_COLS = [
    "recency",
    "frequency",
    "monetary_mean",
    "monetary_total",
    "avg_days_between_orders",
    "num_unique_products",
    "num_unique_countries",
    "weekend_purchase_ratio",
    "return_rate",
    "first_purchase_recency",
]


def load_churn_model(model_path: Path = OUTPUTS_DIR / "churn_model.pkl"):
    """Load the trained churn XGBClassifier Pipeline from disk.

    Parameters
    ----------
    model_path : Path to the saved joblib churn pipeline.

    Returns
    -------
    Fitted sklearn Pipeline, or None if the file does not exist yet.
    """
    if not model_path.exists():
        logger.warning(f"Churn model not found at {model_path}. Run train.py first.")
        return None
    logger.info(f"Loading churn model from {model_path}")
    return joblib.load(model_path)


def load_model(model_path: Path = OUTPUTS_DIR / "model.pkl"):
    """Load the trained sklearn Pipeline from disk.

    Parameters
    ----------
    model_path : Path to the saved joblib pipeline.

    Returns
    -------
    Fitted sklearn Pipeline.
    """
    logger.info(f"Loading model from {model_path}")
    return joblib.load(model_path)


def load_features(
    features_path: Path = OUTPUTS_DIR / "features.parquet",
) -> pd.DataFrame:
    """Load the precomputed customer feature matrix.

    Parameters
    ----------
    features_path : Path to the parquet file written by train.run_training().

    Returns
    -------
    DataFrame indexed by Customer ID.
    """
    return pd.read_parquet(features_path)


def predict_customer(
    customer_id: str | int,
    features: pd.DataFrame,
    pipeline,
) -> Optional[dict]:
    """Predict CLV for a single customer and return their feature values.

    Parameters
    ----------
    customer_id : Customer ID to look up (will be cast to float for index matching).
    features    : Full customer feature DataFrame indexed by Customer ID.
    pipeline    : Fitted sklearn Pipeline.

    Returns
    -------
    Dictionary with 'predicted_clv', 'high_value', and all feature values,
    or None if the customer is not found in the feature matrix.
    """
    try:
        cid = float(customer_id)
    except (ValueError, TypeError):
        return None

    if cid not in features.index:
        return None

    row = features.loc[[cid], FEATURE_COLS]
    pred = float(pipeline.predict(row)[0])
    threshold = float(features["future_revenue"].quantile(0.75))

    result: dict = {
        "predicted_clv": pred,
        "high_value": pred >= threshold,
        "clv_threshold": threshold,
    }
    result.update(row.iloc[0].to_dict())
    return result


def get_shap_explanation(
    customer_id: str | int,
    features: pd.DataFrame,
    pipeline,
) -> Optional[tuple]:
    """Compute SHAP values for a single customer.

    Parameters
    ----------
    customer_id : Customer ID to explain.
    features    : Full customer feature DataFrame indexed by Customer ID.
    pipeline    : Fitted sklearn Pipeline.

    Returns
    -------
    (shap.Explanation, feature_row_df) tuple, or None if customer not found.
    The Explanation is computed on the scaled feature values so that it
    matches the model's internal representation.
    """
    try:
        cid = float(customer_id)
    except (ValueError, TypeError):
        return None

    if cid not in features.index:
        return None

    xgb_model = pipeline.named_steps["model"]
    scaler = pipeline.named_steps["scaler"]

    X = features.loc[[cid], FEATURE_COLS]
    X_scaled = pd.DataFrame(
        scaler.transform(X), columns=FEATURE_COLS, index=X.index
    )

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_scaled)
    return shap_values, X


def simulate_prediction(feature_values: dict[str, float], pipeline) -> float:
    """Predict CLV from a manually specified feature vector.

    Used by the What-If Simulator in the Streamlit app: the user adjusts sliders
    and this function returns the updated CLV in real time.

    Parameters
    ----------
    feature_values : Dict mapping every FEATURE_COLS name to a numeric value.
    pipeline       : Fitted sklearn Pipeline (StandardScaler + XGBRegressor).

    Returns
    -------
    Predicted CLV as a float.
    """
    row = pd.DataFrame([feature_values])[FEATURE_COLS]
    return float(pipeline.predict(row)[0])


def predict_churn_proba(
    customer_id: str | int,
    features: pd.DataFrame,
    churn_pipeline,
) -> Optional[dict]:
    """Predict churn probability for a single customer.

    Parameters
    ----------
    customer_id    : Customer ID to look up.
    features       : Full customer feature DataFrame indexed by Customer ID.
    churn_pipeline : Fitted churn sklearn Pipeline.

    Returns
    -------
    Dict with 'churn_proba' (P(not returning)) and 'return_proba' (P(returning)),
    or None if the customer is not found.
    """
    try:
        cid = float(customer_id)
    except (ValueError, TypeError):
        return None

    if cid not in features.index:
        return None

    row = features.loc[[cid], FEATURE_COLS]
    proba = churn_pipeline.predict_proba(row)[0]
    return_proba = float(proba[1])
    return {"return_proba": return_proba, "churn_proba": 1.0 - return_proba}


def predict_all(features: pd.DataFrame, pipeline) -> pd.DataFrame:
    """Run inference on every customer in the feature matrix.

    Parameters
    ----------
    features : Customer-level feature DataFrame indexed by Customer ID.
    pipeline : Fitted sklearn Pipeline.

    Returns
    -------
    Copy of features with an additional 'predicted_clv' column.
    """
    X = features[FEATURE_COLS]
    result = features.copy()
    result["predicted_clv"] = pipeline.predict(X)
    return result


def assign_rfm_segment(features: pd.DataFrame) -> pd.Series:
    """Assign a simple RFM segment label to each customer based on quartile scores.

    Scoring (1 = worst, 4 = best):
        R: low recency days = 4 (most recent)
        F: high frequency   = 4 (most frequent)
        M: high monetary    = 4 (highest spend)

    Segments
    --------
    Champions   : R=4, F>=3
    Loyal        : F>=3
    At Risk      : R<=2, F>=3
    Promising    : R>=3, F<=2
    Lost         : R<=2, F<=2
    Others       : everything else

    Parameters
    ----------
    features : Customer-level DataFrame with recency, frequency, monetary_total columns.

    Returns
    -------
    Series of segment label strings indexed like features.
    """
    r_score = pd.qcut(features["recency"].rank(method="first"), q=4, labels=[4, 3, 2, 1]).astype(int)
    f_score = pd.qcut(
        features["frequency"].rank(method="first"), q=4, labels=[1, 2, 3, 4]
    ).astype(int)

    def _label(r: int, f: int) -> str:
        if r == 4 and f >= 3:
            return "Champions"
        if f >= 3 and r >= 3:
            return "Loyal"
        if r <= 2 and f >= 3:
            return "At Risk"
        if r >= 3 and f <= 2:
            return "Promising"
        if r <= 2 and f <= 2:
            return "Lost"
        return "Others"

    return pd.Series(
        [_label(r, f) for r, f in zip(r_score, f_score)],
        index=features.index,
        name="rfm_segment",
    )
