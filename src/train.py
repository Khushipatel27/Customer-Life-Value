"""
Model training, evaluation, and SHAP explainability for CLV prediction.

Run from project root:
    python src/train.py
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import pearsonr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
    classification_report,
)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

# Allow running as `python src/train.py` from project root
sys.path.insert(0, str(Path(__file__).parent))
from bgn_model import run_bgn_training
from preprocess import (
    FEATURE_COLS,
    RAW_FILE,
    ROOT_DIR,
    clean_data,
    compute_return_rates,
    engineer_features,
    load_data,
)

logger = logging.getLogger(__name__)

OUTPUTS_DIR = ROOT_DIR / "outputs"
RANDOM_STATE = 42
TARGET_COL = "future_revenue"

_FEATURE_INTERPRETATIONS: dict[str, str] = {
    "recency": "More recent buyers are far more likely to purchase again, directly lifting CLV.",
    "frequency": "Customers who order more often show stronger brand loyalty and higher lifetime spend.",
    "monetary_mean": "A higher average order value signals willingness to pay and premium product affinity.",
    "monetary_total": "Customers with larger historical spend tend to remain high-value over time.",
    "avg_days_between_orders": "Shorter inter-purchase gaps indicate habitual buying behaviour and higher retention.",
    "num_unique_products": "Broad product exploration reflects deeper engagement and cross-sell potential.",
    "num_unique_countries": "Multi-country purchasing often marks wholesale or B2B buyers with high repeat value.",
    "weekend_purchase_ratio": "Purchase-timing patterns capture lifestyle segments with different spend propensity.",
    "return_rate": "Low return rates signal satisfaction; high rates reduce net realised revenue.",
    "first_purchase_recency": "Longer customer tenure correlates with greater cumulative brand trust and CLV.",
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error, ignoring zero actuals to avoid division by zero.

    Parameters
    ----------
    y_true : Ground-truth values.
    y_pred : Model predictions.

    Returns
    -------
    MAPE as a percentage (e.g. 12.3 means 12.3 %).
    """
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, R2, MAPE, and Pearson correlation.

    Parameters
    ----------
    y_true : Ground-truth values.
    y_pred : Model predictions.

    Returns
    -------
    Dictionary of metric name -> value.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    pearson_r, _ = pearsonr(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": mape(y_true, y_pred),
        "Pearson_r": float(pearson_r),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def build_pipeline() -> Pipeline:
    """Create an sklearn Pipeline with StandardScaler and XGBRegressor.

    Returns
    -------
    Unfitted Pipeline.
    """
    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    return Pipeline([("scaler", StandardScaler()), ("model", xgb)])


def train(
    features: pd.DataFrame,
) -> tuple[Pipeline, dict[str, float], np.ndarray, np.ndarray]:
    """Fit the model on an 80/20 split and report test-set metrics plus 5-fold CV RMSE.

    Parameters
    ----------
    features : Customer-level feature matrix from preprocess.engineer_features().

    Returns
    -------
    pipeline : Fitted sklearn Pipeline.
    metrics  : Test-set evaluation metrics.
    y_test   : True labels for the test split.
    y_pred   : Predictions for the test split.
    """
    X = features[FEATURE_COLS]
    y = features[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    logger.info(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # 5-fold cross-validation on training set
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = np.sqrt(
        -cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="neg_mean_squared_error"
        )
    )
    logger.info(f"5-fold CV RMSE: {cv_scores.mean():.2f} +/- {cv_scores.std():.2f}")

    y_pred = pipeline.predict(X_test)
    metrics = evaluate_metrics(y_test.values, y_pred)
    logger.info(f"Test metrics: {metrics}")

    return pipeline, metrics, y_test.values, y_pred


# ---------------------------------------------------------------------------
# SHAP explainability
# ---------------------------------------------------------------------------


def generate_shap_plots(pipeline: Pipeline, X: pd.DataFrame) -> None:
    """Compute SHAP values and save summary, beeswarm, and dependence plots.

    Also prints the top 3 most important features with business interpretations.

    Parameters
    ----------
    pipeline : Fitted sklearn Pipeline (must contain 'scaler' and 'model' steps).
    X        : Feature DataFrame (unscaled) to explain.
    """
    OUTPUTS_DIR.mkdir(exist_ok=True)
    xgb_model = pipeline.named_steps["model"]
    scaler = pipeline.named_steps["scaler"]

    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_scaled)

    # 1. Summary bar plot
    shap.plots.bar(shap_values, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved shap_summary.png")

    # 2. Beeswarm plot
    shap.plots.beeswarm(shap_values, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved shap_beeswarm.png")

    # 3. Dependence plot for the top feature
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top_idx = int(np.argmax(mean_abs))
    top_feature = X.columns[top_idx]
    logger.info(f"Top SHAP feature: {top_feature}")

    shap.plots.scatter(shap_values[:, top_feature], show=False)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "shap_dependence.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved shap_dependence.png")

    # Print top-3 business interpretations
    top3_idx = np.argsort(mean_abs)[::-1][:3]
    print("\nTop 3 CLV drivers:")
    for i in top3_idx:
        feat = X.columns[i]
        interp = _FEATURE_INTERPRETATIONS.get(
            feat, "This feature significantly influences predicted CLV."
        )
        print(f"  {feat} (mean |SHAP| = {mean_abs[i]:.2f}): {interp}")


# ---------------------------------------------------------------------------
# Churn model
# ---------------------------------------------------------------------------


def train_churn_model(
    features: pd.DataFrame,
) -> tuple[Pipeline, dict[str, float]]:
    """Train an XGBoost classifier to predict whether a customer will return.

    Target: will_return = 1 if the customer made any purchase in the prediction
    window (Jan-Mar 2011), else 0.  class imbalance is handled via scale_pos_weight.

    Artefacts saved to outputs/:
        churn_model.pkl, churn_metrics.json,
        churn_predictions.parquet (all customers),
        churn_test_predictions.parquet (test split, for ROC curve in app),
        churn_roc_curve.parquet (fpr/tpr for Plotly ROC chart).

    Parameters
    ----------
    features : Customer-level feature matrix that includes a 'will_return' column.

    Returns
    -------
    (fitted Pipeline, metrics dict)
    """
    X = features[FEATURE_COLS]
    y = features["will_return"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(
        f"Churn train: {len(X_train):,}  |  Test: {len(X_test):,}  |  "
        f"Return rate: {y.mean():.2%}"
    )

    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    xgb_clf = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        eval_metric="auc",
    )
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", xgb_clf)])
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred_cls = pipeline.predict(X_test)

    auc = float(roc_auc_score(y_test, y_proba))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    report = classification_report(y_test, y_pred_cls, output_dict=True)

    metrics: dict[str, float] = {
        "AUC_ROC": auc,
        "Accuracy": float(report["accuracy"]),
        "Precision_returned": float(report.get("1", {}).get("precision", 0.0)),
        "Recall_returned": float(report.get("1", {}).get("recall", 0.0)),
        "F1_returned": float(report.get("1", {}).get("f1-score", 0.0)),
    }
    logger.info(f"Churn model metrics: {metrics}")

    # Save ROC curve data for interactive plot in Streamlit
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_parquet(
        OUTPUTS_DIR / "churn_roc_curve.parquet", index=False
    )

    # Save test-split predictions for additional diagnostics
    pd.DataFrame({"y_test": y_test.values, "y_proba": y_proba}).to_parquet(
        OUTPUTS_DIR / "churn_test_predictions.parquet", index=False
    )

    # Save churn probability for every customer (used by the risk matrix in Tab 2)
    all_proba = pipeline.predict_proba(X)[:, 1]
    pd.DataFrame(
        {"return_proba": all_proba, "churn_proba": 1 - all_proba},
        index=features.index,
    ).to_parquet(OUTPUTS_DIR / "churn_predictions.parquet")
    logger.info("Saved churn_predictions.parquet")

    return pipeline, metrics


# ---------------------------------------------------------------------------
# Artefact persistence
# ---------------------------------------------------------------------------


def save_actual_vs_predicted(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """Save an actual-vs-predicted scatter plot to outputs/.

    Parameters
    ----------
    y_test : True test labels.
    y_pred : Model predictions on the test set.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.4, s=12, color="steelblue")
    lims = [
        min(float(y_test.min()), float(y_pred.min())),
        max(float(y_test.max()), float(y_pred.max())),
    ]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual Revenue (£)")
    ax.set_ylabel("Predicted Revenue (£)")
    ax.set_title("Actual vs. Predicted CLV")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved actual_vs_predicted.png")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_training(raw_path: Path = RAW_FILE) -> None:
    """Full pipeline: preprocess -> train -> evaluate -> save all artefacts.

    Artefacts written to outputs/:
        model.pkl, features.parquet, metrics.json,
        test_predictions.parquet, actual_vs_predicted.png,
        shap_summary.png, shap_beeswarm.png, shap_dependence.png,
        churn_model.pkl, churn_metrics.json, churn_predictions.parquet,
        churn_test_predictions.parquet, churn_roc_curve.parquet,
        bgn_predictions.parquet, bgf_params.json, ggf_params.json

    Parameters
    ----------
    raw_path : Path to the raw Excel file.
    """
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # Run preprocessing steps individually so df_clean is available for BG/NBD
    df_raw = load_data(raw_path)
    return_rates = compute_return_rates(df_raw)
    df_clean = clean_data(df_raw)
    features = engineer_features(df_clean, return_rates)

    pipeline, metrics, y_test, y_pred = train(features)

    print("\n=== Test Set Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Model
    joblib.dump(pipeline, OUTPUTS_DIR / "model.pkl")
    logger.info("Saved model.pkl")

    # Features (used by Streamlit app)
    features.to_parquet(OUTPUTS_DIR / "features.parquet")
    logger.info("Saved features.parquet")

    # Metrics
    with open(OUTPUTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics.json")

    # Test predictions (for actual-vs-predicted plot in Streamlit)
    pd.DataFrame({"y_test": y_test, "y_pred": y_pred}).to_parquet(
        OUTPUTS_DIR / "test_predictions.parquet", index=False
    )
    logger.info("Saved test_predictions.parquet")

    save_actual_vs_predicted(y_test, y_pred)

    # SHAP on the full feature set for representative explanations
    generate_shap_plots(pipeline, features[FEATURE_COLS])

    # Churn model
    print("\n=== Training Churn Model ===")
    churn_pipeline, churn_metrics = train_churn_model(features)
    joblib.dump(churn_pipeline, OUTPUTS_DIR / "churn_model.pkl")
    logger.info("Saved churn_model.pkl")
    with open(OUTPUTS_DIR / "churn_metrics.json", "w") as f:
        json.dump(churn_metrics, f, indent=2)
    logger.info("Saved churn_metrics.json")
    print("\n=== Churn Model Metrics ===")
    for k, v in churn_metrics.items():
        print(f"  {k}: {v:.4f}")

    # BG/NBD + Gamma-Gamma probabilistic CLV
    print("\n=== Training BG/NBD + Gamma-Gamma Model ===")
    bgn_preds = run_bgn_training(df_clean)
    print(f"  BG/NBD mean CLV:   £{bgn_preds['bgnbd_clv'].mean():.2f}")
    print(f"  BG/NBD median CLV: £{bgn_preds['bgnbd_clv'].median():.2f}")
    print(f"  Mean P(alive):     {bgn_preds['bgnbd_prob_alive'].mean():.3f}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run_training()
