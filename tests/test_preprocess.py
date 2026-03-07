"""
Unit tests for src/preprocess.py

Tests cover data cleaning steps and return-rate computation using a small
synthetic DataFrame that does not require the real Excel file.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocess import FEATURE_COLS, clean_data, compute_return_rates


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Minimal raw transaction DataFrame that exercises every cleaning rule."""
    return pd.DataFrame(
        {
            "Invoice": ["I1", "I1", "I2", "I3", "I3", "I4", "I4", "I5"],
            "StockCode": ["A", "A", "B", "C", "C", "D", "D", "E"],
            "Description": ["P1", "P1", "P2", "P3", "P3", "P4", "P4", "P5"],
            # I2 has negative Quantity (return), I3 has zero Price, I5 has null Customer ID
            "Quantity": [2, 2, -1, 3, 3, 5, 5, 1],
            "InvoiceDate": pd.to_datetime(
                [
                    "2010-02-01",
                    "2010-02-01",
                    "2010-04-01",
                    "2010-05-01",
                    "2010-05-01",
                    "2010-06-01",
                    "2010-06-01",
                    "2010-07-01",
                ]
            ),
            "Price": [10.0, 10.0, 5.0, 0.0, 8.0, 6.0, 6.0, 4.0],
            "Customer ID": [
                1001.0, 1001.0, 1002.0, None, 1003.0, 1004.0, 1004.0, None
            ],
            "Country": ["UK", "UK", "UK", "Germany", "UK", "France", "France", "UK"],
        }
    )


# ---------------------------------------------------------------------------
# compute_return_rates
# ---------------------------------------------------------------------------


def test_return_rates_are_between_0_and_1(raw_df):
    rates = compute_return_rates(raw_df)
    assert (rates >= 0).all()
    assert (rates <= 1).all()


def test_return_rate_customer_with_one_return(raw_df):
    rates = compute_return_rates(raw_df)
    # Customer 1002 has exactly 1 row and it has Quantity = -1  ->  rate = 1.0
    assert rates.loc[1002.0] == pytest.approx(1.0)


def test_return_rate_customer_with_no_returns(raw_df):
    rates = compute_return_rates(raw_df)
    # Customer 1001 has 2 rows, both positive  ->  rate = 0.0
    assert rates.loc[1001.0] == pytest.approx(0.0)


def test_return_rates_excludes_null_customer_ids(raw_df):
    rates = compute_return_rates(raw_df)
    assert rates.isna().sum() == 0


# ---------------------------------------------------------------------------
# clean_data
# ---------------------------------------------------------------------------


def test_clean_data_drops_null_customer_id(raw_df):
    df = clean_data(raw_df)
    assert df["Customer ID"].isna().sum() == 0


def test_clean_data_removes_negative_quantity(raw_df):
    df = clean_data(raw_df)
    assert (df["Quantity"] > 0).all()


def test_clean_data_removes_zero_price(raw_df):
    df = clean_data(raw_df)
    assert (df["Price"] > 0).all()


def test_clean_data_removes_duplicates(raw_df):
    df = clean_data(raw_df)
    assert df.duplicated().sum() == 0


def test_clean_data_adds_revenue_column(raw_df):
    df = clean_data(raw_df)
    assert "Revenue" in df.columns
    assert (df["Revenue"] > 0).all()


def test_clean_data_revenue_equals_quantity_times_price(raw_df):
    df = clean_data(raw_df)
    expected = df["Quantity"] * df["Price"]
    pd.testing.assert_series_equal(df["Revenue"], expected, check_names=False)


def test_clean_data_keeps_only_repeat_customers(raw_df):
    """Every remaining customer must have >= 2 distinct invoices."""
    df = clean_data(raw_df)
    invoice_counts = df.groupby("Customer ID")["Invoice"].nunique()
    assert (invoice_counts >= 2).all()


def test_clean_data_output_has_expected_columns(raw_df):
    df = clean_data(raw_df)
    for col in ["Invoice", "Customer ID", "Quantity", "Price", "Revenue", "InvoiceDate"]:
        assert col in df.columns
