# Customer Lifetime Value Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red?logo=streamlit)
![SHAP](https://img.shields.io/badge/SHAP-explainability-brightgreen)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-f7931e?logo=scikit-learn)

## Project Overview

This project predicts the 12-month revenue potential (Customer Lifetime Value) of individual retail customers using the UCI Online Retail II dataset. An XGBoost regression model is trained on 10 RFM-based behavioural features and served through an interactive Streamlit dashboard with SHAP explainability.

---

## Business Problem & Why CLV Matters

Acquiring a new customer costs 5–25× more than retaining an existing one. By accurately estimating each customer's future revenue, businesses can:

- Allocate marketing budgets towards high-value segments
- Trigger early retention campaigns for at-risk customers
- Personalise promotions based on predicted spend tier

A robust CLV model turns historical transaction data into actionable, forward-looking customer intelligence.

---

## Dataset

**UCI Online Retail II** — [https://archive.ics.uci.edu/dataset/502/online+retail+ii](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

- ~1 million rows spanning two years (2009–2011)
- Columns: `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country`
- Download the Excel file and place it at `data/online_retail_ii.xlsx`

---

## Methodology

### Feature Engineering (RFM + Extended)

A **12-month observation window** (Jan 2010 – Dec 2010) is used to build customer-level features:

| Feature | Description |
|---------|-------------|
| `recency` | Days since last purchase |
| `frequency` | Number of unique invoices |
| `monetary_mean` | Average per-invoice revenue |
| `monetary_total` | Total spend in observation window |
| `avg_days_between_orders` | Mean inter-purchase interval |
| `num_unique_products` | Distinct StockCodes purchased |
| `num_unique_countries` | Distinct countries purchased from |
| `weekend_purchase_ratio` | Fraction of orders on Sat/Sun |
| `return_rate` | Ratio of negative-quantity rows |
| `first_purchase_recency` | Days from first purchase to window end |

**Target:** Total revenue in the **3-month prediction window** (Jan 2011 – Mar 2011).

### Model

- **Algorithm:** XGBoost Regressor (`n_estimators=300, max_depth=5, learning_rate=0.05`)
- **Preprocessing:** StandardScaler inside an sklearn Pipeline
- **Validation:** 80/20 train-test split + 5-fold cross-validation

### Explainability

SHAP `TreeExplainer` is used to generate:
- Global feature importance (bar + beeswarm plots)
- Per-customer waterfall explanations in the Streamlit app

---

## Results

| Metric | Value |
|--------|-------|
| MAE (£) | *run `python src/train.py` to populate* |
| RMSE (£) | |
| R² | |
| MAPE (%) | |
| Pearson r | |

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place the dataset
#    Download from https://archive.ics.uci.edu/dataset/502/online+retail+ii
#    and save as data/online_retail_ii.xlsx

# 3. Train the model (generates all outputs/ artefacts)
python src/train.py

# 4. Launch the Streamlit app
streamlit run app/streamlit_app.py

# 5. (Optional) Run the full EDA notebook
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```

---

## Key Insights

1. **Recency is the strongest CLV driver** — customers who purchased recently are far more likely to buy again in the next quarter.
2. **Inter-purchase interval predicts loyalty** — habitual buyers (short gaps between orders) generate disproportionately higher lifetime revenue.
3. **Broad product exploration signals high value** — customers who purchase across many distinct product categories show deeper brand engagement and higher retention.

---

## Future Work

- **BG/NBD probabilistic model** — incorporate the Beta-Geometric/NBD model for non-contractual CLV estimation and churn probability scoring.
- **Customer segmentation with K-Means** — layer unsupervised RFM clusters on top of the predictive model to create actionable marketing personas.
- **Cloud deployment** — containerise the Streamlit app with Docker and deploy to AWS EC2 or Streamlit Community Cloud for live demonstration.

---

## Project Structure

```
customer_life_value/
├── data/
│   └── online_retail_ii.xlsx    # raw data (gitignored)
├── notebooks/
│   └── 01_eda_and_modeling.ipynb
├── src/
│   ├── preprocess.py            # cleaning + feature engineering
│   ├── train.py                 # training, evaluation, SHAP
│   └── predict.py               # inference utilities
├── app/
│   └── streamlit_app.py         # 3-tab Streamlit dashboard
├── outputs/                     # model, plots, metrics (generated)
├── requirements.txt
└── README.md
```
