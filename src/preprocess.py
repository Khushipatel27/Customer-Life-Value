"""
Data loading, cleaning, and feature engineering for CLV prediction.

Observation window: 2010-01-01 to 2010-12-31
Prediction window:  2011-01-01 to 2011-03-31
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_FILE = DATA_DIR / "online_retail_ii.xlsx"

OBSERVATION_START = pd.Timestamp("2010-01-01")
OBSERVATION_END = pd.Timestamp("2010-12-31")
PREDICTION_START = pd.Timestamp("2011-01-01")
PREDICTION_END = pd.Timestamp("2011-03-31")

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


def load_data(path: Path = RAW_FILE) -> pd.DataFrame:
    """Load and concatenate both sheets from the UCI Online Retail II Excel file.

    Parameters
    ----------
    path : Path to the Excel file.

    Returns
    -------
    Combined DataFrame with all transactions and parsed InvoiceDate.
    """
    logger.info(f"Loading data from {path}")
    df_09_10 = pd.read_excel(path, sheet_name="Year 2009-2010", engine="openpyxl")
    df_10_11 = pd.read_excel(path, sheet_name="Year 2010-2011", engine="openpyxl")
    df = pd.concat([df_09_10, df_10_11], ignore_index=True)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    logger.info(
        f"Loaded {len(df):,} rows | "
        f"{df['Customer ID'].nunique():,} unique customers"
    )
    return df


def compute_return_rates(df_raw: pd.DataFrame) -> pd.Series:
    """Compute per-customer return rate from the raw (uncleaned) DataFrame.

    Return rate = (rows where Quantity < 0) / (total rows) per customer,
    computed before cleaning so that cancelled transactions are included.

    Parameters
    ----------
    df_raw : Raw transaction DataFrame as loaded from Excel.

    Returns
    -------
    Series indexed by Customer ID with values in [0, 1].
    """
    df = df_raw.dropna(subset=["Customer ID"])
    total = df.groupby("Customer ID").size()
    returns = df[df["Quantity"] < 0].groupby("Customer ID").size()
    return (returns / total).fillna(0).rename("return_rate")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply sequential data quality filters to the raw transaction log.

    Steps (shape logged after each):
    1. Drop rows with null Customer ID.
    2. Remove returns (Quantity <= 0) and invalid prices (Price <= 0).
    3. Drop exact duplicate rows.
    4. Keep only customers with at least 2 distinct invoices.
    5. Add Revenue column (Quantity * Price).

    Parameters
    ----------
    df : Raw transaction DataFrame.

    Returns
    -------
    Cleaned DataFrame with a Revenue column added.
    """
    logger.info(f"Shape before cleaning: {df.shape}")
    df = df.copy()

    df = df.dropna(subset=["Customer ID"])
    logger.info(f"Shape after dropping null CustomerID:  {df.shape}")

    df = df[df["Quantity"] > 0]
    logger.info(f"Shape after removing Quantity <= 0:    {df.shape}")

    df = df[df["Price"] > 0]
    logger.info(f"Shape after removing Price <= 0:       {df.shape}")

    df = df.drop_duplicates()
    logger.info(f"Shape after removing duplicates:       {df.shape}")

    invoice_counts = df.groupby("Customer ID")["Invoice"].nunique()
    valid_customers = invoice_counts[invoice_counts >= 2].index
    df = df[df["Customer ID"].isin(valid_customers)]
    logger.info(f"Shape after filtering to >=2 invoices: {df.shape}")

    df["Revenue"] = df["Quantity"] * df["Price"]
    return df


def engineer_features(
    df_clean: pd.DataFrame,
    return_rates: pd.Series,
) -> pd.DataFrame:
    """Build a customer-level feature matrix from the observation window.

    Only customers present in the observation window (Jan-Dec 2010) are included.
    The target (future_revenue) is summed from the prediction window (Jan-Mar 2011);
    customers with no prediction-window activity receive a target of 0.

    Features engineered
    -------------------
    recency                  Days since last purchase at observation window end.
    frequency                Number of unique invoices in observation window.
    monetary_mean            Average per-invoice revenue.
    monetary_total           Total revenue in observation window.
    avg_days_between_orders  Mean gap (days) between consecutive invoices.
    num_unique_products      Distinct StockCodes purchased.
    num_unique_countries     Distinct countries the customer purchased from.
    weekend_purchase_ratio   Fraction of invoices placed on Saturday or Sunday.
    return_rate              Fraction of all raw rows with Quantity < 0.
    first_purchase_recency   Days from earliest purchase to observation window end.

    Targets / metadata
    ------------------
    future_revenue       Total spend in prediction window.
    high_value_customer  1 if future_revenue >= 75th percentile, else 0.
    primary_country      Most frequent country (used by the Streamlit filter).

    Parameters
    ----------
    df_clean    : Cleaned transaction DataFrame with a Revenue column.
    return_rates: Per-customer return rates from compute_return_rates().

    Returns
    -------
    DataFrame indexed by Customer ID.
    """
    obs = df_clean[
        (df_clean["InvoiceDate"] >= OBSERVATION_START)
        & (df_clean["InvoiceDate"] <= OBSERVATION_END)
    ].copy()

    pred = df_clean[
        (df_clean["InvoiceDate"] >= PREDICTION_START)
        & (df_clean["InvoiceDate"] <= PREDICTION_END)
    ]

    obs_customers = obs["Customer ID"].unique()
    logger.info(f"Customers in observation window: {len(obs_customers):,}")

    # Per-invoice aggregations (avoids slow row-level groupby-apply)
    order_agg = (
        obs.groupby(["Customer ID", "Invoice"])
        .agg(revenue=("Revenue", "sum"), date=("InvoiceDate", "min"))
        .reset_index()
    )

    last_purchase = order_agg.groupby("Customer ID")["date"].max()
    first_purchase = order_agg.groupby("Customer ID")["date"].min()

    recency = (OBSERVATION_END - last_purchase).dt.days.rename("recency")
    first_purchase_recency = (OBSERVATION_END - first_purchase).dt.days.rename(
        "first_purchase_recency"
    )
    frequency = order_agg.groupby("Customer ID")["Invoice"].count().rename("frequency")
    monetary_mean = (
        order_agg.groupby("Customer ID")["revenue"].mean().rename("monetary_mean")
    )
    monetary_total = (
        order_agg.groupby("Customer ID")["revenue"].sum().rename("monetary_total")
    )

    # Vectorised inter-purchase interval
    order_sorted = order_agg.sort_values(["Customer ID", "date"])
    order_sorted["days_diff"] = (
        order_sorted.groupby("Customer ID")["date"].diff().dt.days
    )
    avg_days = (
        order_sorted.groupby("Customer ID")["days_diff"]
        .mean()
        .fillna(0)
        .rename("avg_days_between_orders")
    )

    num_unique_products = (
        obs.groupby("Customer ID")["StockCode"].nunique().rename("num_unique_products")
    )
    num_unique_countries = (
        obs.groupby("Customer ID")["Country"].nunique().rename("num_unique_countries")
    )

    obs["is_weekend"] = obs["InvoiceDate"].dt.dayofweek >= 5
    weekend_ratio = (
        obs.groupby("Customer ID")["is_weekend"]
        .mean()
        .rename("weekend_purchase_ratio")
    )

    # Primary country for Streamlit dropdown filter
    primary_country = (
        obs.groupby("Customer ID")["Country"]
        .agg(lambda x: x.mode().iloc[0])
        .rename("primary_country")
    )

    future_revenue = (
        pred[pred["Customer ID"].isin(obs_customers)]
        .groupby("Customer ID")["Revenue"]
        .sum()
        .rename("future_revenue")
    )

    features = pd.concat(
        [
            recency,
            frequency,
            monetary_mean,
            monetary_total,
            avg_days,
            num_unique_products,
            num_unique_countries,
            weekend_ratio,
            return_rates.reindex(obs_customers),
            first_purchase_recency,
            primary_country,
        ],
        axis=1,
    )

    features = features.loc[obs_customers].copy()
    features["future_revenue"] = future_revenue.reindex(features.index).fillna(0)
    features["high_value_customer"] = (
        features["future_revenue"] >= features["future_revenue"].quantile(0.75)
    ).astype(int)

    # Churn target: 1 = customer made at least one purchase in prediction window
    features["will_return"] = (features["future_revenue"] > 0).astype(int)

    features = features.dropna(subset=FEATURE_COLS)
    logger.info(f"Final feature matrix: {features.shape}")
    return features


def run_preprocessing(raw_path: Path = RAW_FILE) -> pd.DataFrame:
    """End-to-end pipeline: load -> compute return rates -> clean -> engineer features.

    Parameters
    ----------
    raw_path : Path to the raw Excel file.

    Returns
    -------
    Customer-level feature DataFrame ready for modelling.
    """
    df_raw = load_data(raw_path)
    return_rates = compute_return_rates(df_raw)
    df_clean = clean_data(df_raw)
    features = engineer_features(df_clean, return_rates)
    return features


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    features = run_preprocessing()
    out_path = ROOT_DIR / "outputs" / "features.parquet"
    out_path.parent.mkdir(exist_ok=True)
    features.to_parquet(out_path)
    logger.info(f"Saved features to {out_path}")
    print(features.describe())
