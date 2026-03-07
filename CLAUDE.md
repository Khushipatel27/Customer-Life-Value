# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Customer Lifetime Value (CLV) prediction using the UCI Online Retail II dataset. Predicts 3-month forward revenue per customer (observation window Jan–Dec 2010, prediction window Jan–Mar 2011) using XGBoost on 10 RFM-based features, with SHAP explainability and a Streamlit dashboard.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train model and generate all outputs/ artefacts
python src/train.py

# Launch the Streamlit dashboard (requires trained model)
streamlit run app/streamlit_app.py

# Run the EDA + modelling notebook
jupyter notebook notebooks/01_eda_and_modeling.ipynb

# Run just preprocessing to inspect the feature matrix
python src/preprocess.py
```

## Architecture

```
data/                            # raw Excel file — gitignored
src/
  preprocess.py                  # load → clean → feature engineering → parquet
  train.py                       # train XGBoost, evaluate, SHAP plots, save artefacts
  predict.py                     # inference utilities + RFM segmentation (used by app)
app/
  streamlit_app.py               # 3-tab dashboard (imports from src/)
notebooks/
  01_eda_and_modeling.ipynb      # end-to-end EDA + modelling, runs top-to-bottom
outputs/                         # model.pkl, features.parquet, metrics.json,
                                 # test_predictions.parquet, PNG plots (generated)
```

## Data Pipeline

UCI Online Retail II has columns: `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country` (note: "Price" not "UnitPrice", "Customer ID" with a space).

1. Load both Excel sheets (`Year 2009-2010`, `Year 2010-2011`) and concatenate
2. Compute `return_rate` from raw data **before** cleaning (includes cancellations)
3. Clean: drop null `Customer ID`, `Quantity <= 0`, `Price <= 0`, duplicates, keep ≥2 invoices
4. Engineer 10 customer-level features from the **observation window** (Jan–Dec 2010)
5. Target `future_revenue` = total spend in the **prediction window** (Jan–Mar 2011)

## Model

- Pipeline: `StandardScaler` → `XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)`
- 80/20 split + 5-fold CV; metrics: MAE, RMSE, R², MAPE, Pearson r
- Artefacts saved: `outputs/model.pkl` (joblib), `outputs/features.parquet`, `outputs/metrics.json`

## Streamlit App Tabs

- **Tab 1 — Customer Lookup:** CustomerID text input → predicted CLV + SHAP waterfall
- **Tab 2 — Segment Explorer:** frequency vs. predicted CLV scatter coloured by RFM segment, country dropdown filter, segment stats table
- **Tab 3 — Model Performance:** metric cards + actual-vs-predicted plot + SHAP summary bar

## Code Conventions

- All functions have docstrings and type hints
- Use `pathlib.Path` — no hardcoded string paths
- Use `logging` in `src/` files (not bare `print`)
- `requirements.txt` uses pinned versions
- The notebook must run top-to-bottom without errors
- `streamlit_app.py` adds `src/` to `sys.path` at import time for portability
