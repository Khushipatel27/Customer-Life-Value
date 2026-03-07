"""
BG/NBD + Gamma-Gamma probabilistic CLV model.

The BG/NBD (Beta-Geometric/Negative Binomial Distribution) model estimates:
  - P(alive)  : probability a customer is still active
  - Expected future transactions over a given time horizon

The Gamma-Gamma model estimates:
  - Expected average transaction value per customer (repeat purchasers only)

Combined: probabilistic CLV = expected purchases × expected value per transaction.
This serves as a domain-appropriate baseline to compare against XGBoost.

Run standalone:
    python src/bgn_model.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = ROOT_DIR / "outputs"

OBSERVATION_END = pd.Timestamp("2010-12-31")
PREDICTION_MONTHS = 3  # match XGBoost prediction window


def build_rfm_summary(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Build the customer-level RFM summary table required by lifetimes.

    Aggregates to invoice level first so that multiple line-items in one
    invoice count as a single transaction.  Only transactions up to
    OBSERVATION_END are used.

    Parameters
    ----------
    df_clean : Cleaned transaction DataFrame with Revenue and InvoiceDate columns.

    Returns
    -------
    DataFrame indexed by Customer ID with columns:
        frequency      Number of repeat transactions (total transactions - 1).
        recency        Days between first and last purchase.
        T              Days from first purchase to observation end.
        monetary_value Mean per-invoice revenue across repeat transactions.
    """
    obs_df = df_clean[df_clean["InvoiceDate"] <= OBSERVATION_END].copy()

    # One row per invoice
    invoice_df = (
        obs_df.groupby(["Customer ID", "Invoice"])
        .agg(date=("InvoiceDate", "min"), revenue=("Revenue", "sum"))
        .reset_index()
    )

    rfm = summary_data_from_transaction_data(
        invoice_df,
        customer_id_col="Customer ID",
        datetime_col="date",
        monetary_value_col="revenue",
        observation_period_end=OBSERVATION_END,
        freq="D",
    )
    logger.info(
        f"RFM summary: {len(rfm):,} customers | "
        f"repeat purchasers (freq > 0): {(rfm['frequency'] > 0).sum():,}"
    )
    return rfm


def train_bgnbd(rfm: pd.DataFrame) -> BetaGeoFitter:
    """Fit a BG/NBD model on the RFM summary.

    Parameters
    ----------
    rfm : Output of build_rfm_summary().

    Returns
    -------
    Fitted BetaGeoFitter.
    """
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(rfm["frequency"], rfm["recency"], rfm["T"])
    logger.info(f"BG/NBD fitted — params: {dict(bgf.params_)}")
    return bgf


def train_gamma_gamma(rfm: pd.DataFrame) -> GammaGammaFitter:
    """Fit a Gamma-Gamma model on repeat purchasers (frequency > 0) only.

    The Gamma-Gamma model is undefined for customers with no repeat purchases
    because there is no second transaction to form a monetary-value distribution.

    Parameters
    ----------
    rfm : Output of build_rfm_summary().

    Returns
    -------
    Fitted GammaGammaFitter.
    """
    repeat = rfm[rfm["frequency"] > 0]
    ggf = GammaGammaFitter(penalizer_coef=0.001)
    ggf.fit(repeat["frequency"], repeat["monetary_value"])
    logger.info(f"Gamma-Gamma fitted — params: {dict(ggf.params_)}")
    return ggf


def predict_clv(
    bgf: BetaGeoFitter,
    ggf: GammaGammaFitter,
    rfm: pd.DataFrame,
    months: int = PREDICTION_MONTHS,
    discount_rate: float = 0.01,
) -> pd.DataFrame:
    """Compute probabilistic CLV, expected purchases, and P(alive) per customer.

    For one-time buyers (frequency = 0) the Gamma-Gamma model cannot be applied;
    their CLV falls back to expected_purchases × global average monetary value.

    Parameters
    ----------
    bgf           : Fitted BetaGeoFitter.
    ggf           : Fitted GammaGammaFitter.
    rfm           : Output of build_rfm_summary().
    months        : Forecast horizon in calendar months.
    discount_rate : Monthly discount rate for present-value adjustment.

    Returns
    -------
    DataFrame indexed by Customer ID with columns:
        bgnbd_expected_purchases  Expected number of transactions in [0, months].
        bgnbd_prob_alive          Probability the customer is still active.
        bgnbd_clv                 Probabilistic CLV over the forecast horizon.
    """
    days = months * 30

    expected_purchases = pd.Series(
        np.asarray(
            bgf.conditional_expected_number_of_purchases_up_to_time(
                days, rfm["frequency"], rfm["recency"], rfm["T"]
            )
        ),
        index=rfm.index,
    )
    prob_alive = pd.Series(
        np.asarray(
            bgf.conditional_probability_alive(
                rfm["frequency"], rfm["recency"], rfm["T"]
            )
        ),
        index=rfm.index,
    )

    repeat_mask = rfm["frequency"] > 0
    clv = pd.Series(np.nan, index=rfm.index, dtype=float)

    if repeat_mask.any():
        clv_vals = ggf.customer_lifetime_value(
            bgf,
            rfm.loc[repeat_mask, "frequency"],
            rfm.loc[repeat_mask, "recency"],
            rfm.loc[repeat_mask, "T"],
            rfm.loc[repeat_mask, "monetary_value"],
            time=months,
            discount_rate=discount_rate,
            freq="D",
        )
        clv.loc[repeat_mask] = clv_vals.values

    # One-time buyers: expected purchases × global average order value
    global_avg = float(
        rfm.loc[repeat_mask, "monetary_value"].mean()
    ) if repeat_mask.any() else 0.0
    clv.loc[~repeat_mask] = expected_purchases.loc[~repeat_mask] * global_avg

    result = pd.DataFrame(
        {
            "bgnbd_expected_purchases": np.asarray(expected_purchases),
            "bgnbd_prob_alive": np.asarray(prob_alive),
            "bgnbd_clv": clv.to_numpy(),
        },
        index=rfm.index,
    )
    logger.info(
        f"BG/NBD CLV — mean: £{result['bgnbd_clv'].mean():.2f} | "
        f"median: £{result['bgnbd_clv'].median():.2f}"
    )
    return result


def run_bgn_training(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Full BG/NBD + Gamma-Gamma pipeline.

    BetaGeoFitter/GammaGammaFitter objects cannot be pickled (they contain
    internal lambda functions), so model parameters are persisted as JSON
    instead. The Streamlit app only requires bgn_predictions.parquet.

    Artefacts saved to outputs/:
        bgn_predictions.parquet  Per-customer probabilistic CLV predictions.
        bgf_params.json          BG/NBD fitted parameter values.
        ggf_params.json          Gamma-Gamma fitted parameter values.

    Parameters
    ----------
    df_clean : Cleaned transaction DataFrame from preprocess.clean_data().

    Returns
    -------
    Predictions DataFrame indexed by Customer ID.
    """
    OUTPUTS_DIR.mkdir(exist_ok=True)
    rfm = build_rfm_summary(df_clean)
    bgf = train_bgnbd(rfm)
    ggf = train_gamma_gamma(rfm)
    predictions = predict_clv(bgf, ggf, rfm)

    predictions.to_parquet(OUTPUTS_DIR / "bgn_predictions.parquet")

    # Save params as JSON (models contain lambdas that can't be pickled)
    with open(OUTPUTS_DIR / "bgf_params.json", "w") as f:
        json.dump({k: float(v) for k, v in bgf.params_.items()}, f, indent=2)
    with open(OUTPUTS_DIR / "ggf_params.json", "w") as f:
        json.dump({k: float(v) for k, v in ggf.params_.items()}, f, indent=2)

    logger.info("Saved bgn_predictions.parquet, bgf_params.json, ggf_params.json")
    return predictions


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    sys.path.insert(0, str(Path(__file__).parent))
    from preprocess import RAW_FILE, load_data, clean_data, compute_return_rates  # noqa: E402

    df_raw = load_data(RAW_FILE)
    df_clean = clean_data(df_raw)
    preds = run_bgn_training(df_clean)
    print(preds.describe())
