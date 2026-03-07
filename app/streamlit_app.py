"""
Streamlit dashboard for CLV prediction.

Tabs
----
1. Customer Lookup   – CLV + churn probability, feature comparison, What-If simulator, SHAP waterfall
2. Segment Explorer  – scatter, donut, CLV×Churn risk matrix, country/segment filters, CSV download
3. Model Performance – CLV model metrics + interactive plots, Churn model metrics + ROC curve

Run:
    streamlit run app/streamlit_app.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from predict import (  # noqa: E402
    assign_rfm_segment,
    get_shap_explanation,
    load_churn_model,
    load_features,
    load_model,
    predict_all,
    predict_churn_proba,
    predict_customer,
    simulate_prediction,
)

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CLV Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="metric-container"] {
        background-color: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetricValue"] { font-size: 1.35rem; }
    div[data-testid="stExpander"] { border: 1px solid #2d3250; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading CLV model…")
def _load_model():
    return load_model()


@st.cache_resource(show_spinner="Loading churn model…")
def _load_churn_model():
    return load_churn_model()


@st.cache_data(show_spinner="Loading features…")
def _load_features() -> pd.DataFrame:
    return load_features()


@st.cache_data(show_spinner="Running predictions…")
def _build_all_predictions() -> pd.DataFrame:
    features = _load_features()
    pipeline = _load_model()
    df = predict_all(features, pipeline)
    df["rfm_segment"] = assign_rfm_segment(features)
    return df


@st.cache_data(show_spinner="Loading churn predictions…")
def _load_churn_predictions() -> pd.DataFrame | None:
    p = OUTPUTS_DIR / "churn_predictions.parquet"
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data(show_spinner="Loading BG/NBD predictions…")
def _load_bgn_predictions() -> pd.DataFrame | None:
    p = OUTPUTS_DIR / "bgn_predictions.parquet"
    return pd.read_parquet(p) if p.exists() else None


# ---------------------------------------------------------------------------
# Artefact guard
# ---------------------------------------------------------------------------

_missing = [
    p
    for p in [
        OUTPUTS_DIR / "model.pkl",
        OUTPUTS_DIR / "features.parquet",
        OUTPUTS_DIR / "metrics.json",
    ]
    if not p.exists()
]
if _missing:
    st.error(
        "**Required artefacts not found.** Run `python src/train.py` first.\n\n"
        f"Missing: `{'`, `'.join(p.name for p in _missing)}`"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Global data
# ---------------------------------------------------------------------------

features = _load_features()
pipeline = _load_model()
churn_pipeline = _load_churn_model()
all_preds = _build_all_predictions()
churn_preds = _load_churn_predictions()
bgn_preds = _load_bgn_predictions()

with open(OUTPUTS_DIR / "metrics.json") as f:
    metrics = json.load(f)

churn_metrics: dict = {}
if (OUTPUTS_DIR / "churn_metrics.json").exists():
    with open(OUTPUTS_DIR / "churn_metrics.json") as f:
        churn_metrics = json.load(f)

_threshold = float(features["future_revenue"].quantile(0.75))
_valid_ids = sorted(features.index.astype(int).tolist())

_FEATURE_LABELS: dict[str, tuple[str, str]] = {
    "recency": ("Recency (days)", "Days since last purchase at window end"),
    "frequency": ("Frequency (invoices)", "Number of unique orders placed"),
    "monetary_mean": ("Avg Order Value (£)", "Mean revenue per invoice"),
    "monetary_total": ("Total Spend (£)", "Cumulative spend in observation window"),
    "avg_days_between_orders": ("Avg Days Between Orders", "Mean inter-purchase gap in days"),
    "num_unique_products": ("Unique Products", "Distinct StockCodes purchased"),
    "num_unique_countries": ("Countries", "Distinct countries ordered from"),
    "weekend_purchase_ratio": ("Weekend Ratio", "Fraction of orders placed on Sat/Sun"),
    "return_rate": ("Return Rate", "Fraction of all raw rows with Quantity < 0"),
    "first_purchase_recency": ("Tenure (days)", "Days from first purchase to window end"),
}

_SEGMENT_COLORS = {
    "Champions": "#2ecc71",
    "Loyal": "#3498db",
    "At Risk": "#e67e22",
    "Promising": "#9b59b6",
    "Lost": "#e74c3c",
    "Others": "#95a5a6",
}

# Merge churn and BG/NBD predictions into all_preds if available
if churn_preds is not None:
    all_preds = all_preds.join(churn_preds[["churn_proba", "return_proba"]], how="left")
if bgn_preds is not None:
    all_preds = all_preds.join(
        bgn_preds[["bgnbd_clv", "bgnbd_prob_alive", "bgnbd_expected_purchases"]],
        how="left",
    )

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("CLV Dashboard")
    st.caption("UCI Online Retail II · XGBoost + SHAP")
    st.divider()

    st.subheader("Dataset Overview")
    st.metric("Total Customers", f"{len(all_preds):,}")
    high_val_n = int(all_preds["predicted_clv"].ge(_threshold).sum())
    st.metric(
        "High-Value (top 25%)",
        f"{high_val_n:,}",
        delta=f"{high_val_n / len(all_preds) * 100:.1f}% of base",
    )
    st.metric("Avg Predicted CLV", f"£{all_preds['predicted_clv'].mean():,.0f}")
    st.metric("Median Predicted CLV", f"£{all_preds['predicted_clv'].median():,.0f}")

    if churn_preds is not None:
        st.divider()
        st.subheader("Churn Overview")
        at_risk_pct = float((all_preds["churn_proba"] > 0.5).mean() * 100)
        st.metric("Churn Risk > 50%", f"{at_risk_pct:.1f}% of customers")

    st.divider()
    st.subheader("Model Quality")
    st.metric("CLV R²", f"{metrics['R2']:.3f}")
    st.metric("CLV RMSE (£)", f"{metrics['RMSE']:,.0f}")
    if churn_metrics:
        st.metric("Churn AUC-ROC", f"{churn_metrics.get('AUC_ROC', 0):.3f}")

    st.divider()
    with st.expander("How to use"):
        st.markdown(
            """
**Tab 1 — Customer Lookup**
Enter any Customer ID to see predicted CLV, churn risk, an action recommendation,
a feature comparison vs the average customer, a **What-If simulator** to explore
hypotheticals, and a SHAP waterfall explaining the prediction.

**Tab 2 — Segment Explorer**
Browse all customers split by RFM segment. Filter by country, CLV range and segment.
The **CLV × Churn Risk Matrix** maps every customer into one of four action quadrants.
Download filtered data as CSV.

**Tab 3 — Model Performance**
Interactive actual-vs-predicted scatter, residuals histogram, CLV distribution,
and churn model ROC curve + metrics.
            """
        )

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("Customer Lifetime Value Prediction")
st.caption(
    "Observation window: Jan–Dec 2010  ·  Prediction window: Jan–Mar 2011  ·  "
    "UCI Online Retail II"
)

st.markdown(
    """
**Customer Lifetime Value (CLV)** is the total revenue a business can expect from a single customer
over a future period. Knowing each customer's CLV allows marketing, sales, and retention teams to
decide *who* to invest in, *how much* to spend, and *what action* to take — before revenue is lost.

This dashboard combines two complementary models:
- **XGBoost** — a gradient-boosted regression model trained on 10 RFM behavioural features
- **BG/NBD + Gamma-Gamma** — the industry-standard probabilistic model that estimates how likely
  a customer is still active and how much they are likely to spend

Use the tabs below to look up individual customers, explore segments, or evaluate model performance.
"""
)
st.divider()

tab1, tab2, tab3 = st.tabs(
    ["🔍 Customer Lookup", "📊 Segment Explorer", "📈 Model Performance"]
)

# ===========================================================================
# Tab 1 — Customer Lookup + What-If Simulator
# ===========================================================================

with tab1:
    st.subheader("Customer Lookup")
    st.markdown(
        "Enter a **Customer ID** from the 2010 observation window to instantly see their "
        "predicted 3-month revenue, churn risk, where they sit in the customer base, and "
        "a full SHAP explanation of *why* the model made that prediction. "
        "Use the **What-If Simulator** below to explore how changing their behaviour "
        "would affect their value."
    )

    col_input, col_btn = st.columns([3, 1])
    with col_input:
        cid_raw = st.text_input(
            "Customer ID",
            value=st.session_state.get("cid_input", ""),
            placeholder="e.g. 12347",
            help="Enter a Customer ID from the 2010 observation window.",
        )
    with col_btn:
        st.write("")
        st.write("")
        if st.button("🎲 Random Customer", use_container_width=True):
            st.session_state["cid_input"] = str(np.random.choice(_valid_ids))
            st.rerun()

    cid_input: str = st.session_state.get("cid_input", cid_raw) if not cid_raw else cid_raw

    if cid_input.strip():
        result = predict_customer(cid_input.strip(), features, pipeline)

        if result is None:
            st.warning(
                f"Customer **{cid_input}** not found in the observation window "
                "(Jan–Dec 2010). Try another ID or click **Random Customer**."
            )
        else:
            pred_clv = result["predicted_clv"]
            clv_pct = float((all_preds["predicted_clv"] < pred_clv).mean() * 100)

            # Churn info
            churn_info: dict | None = None
            if churn_pipeline is not None:
                churn_info = predict_churn_proba(cid_input.strip(), features, churn_pipeline)

            # RFM segment
            try:
                cid_float = float(cid_input.strip())
                segment = (
                    all_preds.loc[cid_float, "rfm_segment"]
                    if cid_float in all_preds.index
                    else "N/A"
                )
            except (ValueError, KeyError):
                segment = "N/A"

            # ----------------------------------------------------------------
            # Metric row
            # ----------------------------------------------------------------
            cols = st.columns(5)
            cols[0].metric(
                "Predicted CLV (3 months)",
                f"£{pred_clv:,.2f}",
                delta="Top 25%" if result["high_value"] else "Bottom 75%",
                delta_color="normal" if result["high_value"] else "inverse",
            )
            cols[1].metric("CLV Percentile", f"{clv_pct:.0f}th")
            cols[2].metric("RFM Segment", segment)
            if churn_info:
                churn_pct = churn_info["churn_proba"] * 100
                cols[3].metric(
                    "Churn Probability",
                    f"{churn_pct:.1f}%",
                    delta="High Risk" if churn_pct > 50 else "Low Risk",
                    delta_color="inverse" if churn_pct > 50 else "normal",
                )
                cols[4].metric(
                    "Return Probability",
                    f"{churn_info['return_proba']*100:.1f}%",
                )
            else:
                cols[3].metric("High-Value Threshold", f"£{_threshold:,.2f}")

            # CLV percentile bar
            st.markdown(f"**CLV position in customer base** — {clv_pct:.0f}th percentile")
            st.progress(clv_pct / 100)

            with st.expander("📖 How to read these metrics"):
                st.markdown(
                    """
| Metric | What it means |
|--------|---------------|
| **Predicted CLV** | The model's estimate of how much revenue this customer will generate in the next 3 months (Jan–Mar 2011). |
| **CLV Percentile** | How this customer ranks against all others — 90th percentile means they are more valuable than 90% of the base. |
| **High-Value Threshold** | The 75th-percentile CLV value. Customers above this line are classified as *High Value* and deserve priority treatment. |
| **RFM Segment** | A rule-based label derived from Recency, Frequency, and Monetary scores. Champions are the best customers; Lost have not bought recently and spend little. |
| **Churn Probability** | The XGBoost classifier's estimate of the chance this customer will **not** make any purchase in the next 3 months. Above 50% = at risk. |
| **Return Probability** | The complement of churn — how likely the customer is to come back. |
                    """
                )

            # ----------------------------------------------------------------
            # Action recommendation
            # ----------------------------------------------------------------
            if churn_info:
                high_clv = result["high_value"]
                high_churn = churn_info["churn_proba"] > 0.5
                if high_clv and not high_churn:
                    action_color, action_icon, action_text = (
                        "green", "👑",
                        "**VIP Customer** — High value, low churn risk. "
                        "Reward with loyalty benefits to maintain engagement.",
                    )
                elif high_clv and high_churn:
                    action_color, action_icon, action_text = (
                        "red", "🚨",
                        "**Save Now** — High value but at serious churn risk. "
                        "Trigger an immediate personalised retention campaign.",
                    )
                elif not high_clv and not high_churn:
                    action_color, action_icon, action_text = (
                        "blue", "🌱",
                        "**Nurture** — Loyal but low-spend. "
                        "Focus on upsell and cross-sell to grow their value.",
                    )
                else:
                    action_color, action_icon, action_text = (
                        "gray", "💤",
                        "**Low Priority** — Low value and likely to churn. "
                        "Re-engage only with low-cost automated campaigns.",
                    )
                st.markdown(
                    f"**Recommended Action** {action_icon}",
                )
                st.info(action_text)

            # ----------------------------------------------------------------
            # BG/NBD Probabilistic Model Comparison
            # ----------------------------------------------------------------
            if bgn_preds is not None:
                try:
                    cid_float = float(cid_input.strip())
                    bgn_row = bgn_preds.loc[cid_float] if cid_float in bgn_preds.index else None
                except (ValueError, KeyError):
                    bgn_row = None

                with st.expander("🔬 BG/NBD Probabilistic Model Comparison", expanded=False):
                    st.caption(
                        "**XGBoost** learns from RFM features via gradient boosting. "
                        "**BG/NBD + Gamma-Gamma** uses a statistical model of buy-till-you-die "
                        "behaviour — the industry standard for non-contractual CLV. "
                        "Comparing both gives a fuller picture of each customer's value."
                    )
                    if bgn_row is not None:
                        bc1, bc2, bc3, bc4 = st.columns(4)
                        bc1.metric(
                            "XGBoost CLV",
                            f"£{pred_clv:,.2f}",
                            help="Gradient boosted regression on RFM features.",
                        )
                        bc2.metric(
                            "BG/NBD CLV",
                            f"£{bgn_row['bgnbd_clv']:,.2f}",
                            delta=f"{'+'if bgn_row['bgnbd_clv'] >= pred_clv else ''}{bgn_row['bgnbd_clv'] - pred_clv:,.2f} vs XGBoost",
                            delta_color="off",
                            help="Gamma-Gamma × BG/NBD expected purchases.",
                        )
                        bc3.metric(
                            "P(Alive)",
                            f"{bgn_row['bgnbd_prob_alive']*100:.1f}%",
                            help="Probability this customer is still an active buyer.",
                        )
                        bc4.metric(
                            "Expected Purchases",
                            f"{bgn_row['bgnbd_expected_purchases']:.2f}",
                            help="BG/NBD expected number of orders in the next 3 months.",
                        )

                        # Mini gauge bar for P(alive)
                        st.markdown(
                            f"**P(Alive) — likelihood customer is still active:** "
                            f"{bgn_row['bgnbd_prob_alive']*100:.1f}%"
                        )
                        st.progress(float(bgn_row["bgnbd_prob_alive"]))
                    else:
                        st.info("Customer not found in BG/NBD predictions.")

            st.divider()

            # ----------------------------------------------------------------
            # Feature comparison vs population
            # ----------------------------------------------------------------
            st.subheader("Feature Profile vs. Population Average")
            st.caption(
                "Blue bar = this customer · Orange marker = population average. "
                "Values normalised to [0, 1] using the 5th–95th percentile range."
            )

            feat_rows = []
            for col, (label, desc) in _FEATURE_LABELS.items():
                cust_val = float(result[col])
                pop_val = float(features[col].mean())
                lo = float(features[col].quantile(0.05))
                hi = float(features[col].quantile(0.95))
                rng = hi - lo if hi != lo else 1.0
                feat_rows.append(
                    {
                        "Feature": label,
                        "Customer (norm)": float(np.clip((cust_val - lo) / rng, 0, 1)),
                        "Pop Avg (norm)": float(np.clip((pop_val - lo) / rng, 0, 1)),
                        "Raw Value": round(cust_val, 2),
                        "Pop Avg (raw)": round(pop_val, 2),
                    }
                )
            feat_df = pd.DataFrame(feat_rows)

            fig_compare = go.Figure()
            fig_compare.add_trace(
                go.Bar(
                    y=feat_df["Feature"],
                    x=feat_df["Customer (norm)"],
                    orientation="h",
                    name="This Customer",
                    marker_color="steelblue",
                    hovertemplate="<b>%{y}</b><br>Customer: %{customdata}<extra></extra>",
                    customdata=feat_df["Raw Value"],
                )
            )
            fig_compare.add_trace(
                go.Scatter(
                    y=feat_df["Feature"],
                    x=feat_df["Pop Avg (norm)"],
                    mode="markers",
                    name="Population Avg",
                    marker=dict(
                        color="orange", size=10, symbol="line-ns-open", line_width=2
                    ),
                    hovertemplate="<b>%{y}</b><br>Pop Avg: %{customdata}<extra></extra>",
                    customdata=feat_df["Pop Avg (raw)"],
                )
            )
            fig_compare.update_layout(
                height=370,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis_title="Normalised Score (0 – 1)",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_compare, use_container_width=True)

            with st.expander("View raw feature values"):
                tbl = feat_df[["Feature", "Raw Value", "Pop Avg (raw)"]].copy()
                tbl.columns = ["Feature", "This Customer", "Population Avg"]
                st.dataframe(tbl, use_container_width=True, hide_index=True)

            st.divider()

            # ----------------------------------------------------------------
            # What-If Simulator
            # ----------------------------------------------------------------
            st.subheader("What-If Simulator")
            st.caption(
                "Adjust any feature below to see how the predicted CLV changes in real time. "
                "Sliders start at this customer's actual values."
            )

            # Slider bounds: 5th–95th percentile of each feature.
            # If both are equal (low-variance feature), widen by ±1 so the slider is valid.
            bounds = {}
            for col in _FEATURE_LABELS:
                lo = float(features[col].quantile(0.05))
                hi = float(features[col].quantile(0.95))
                if hi <= lo:
                    lo = float(features[col].min())
                    hi = float(features[col].max())
                if hi <= lo:
                    hi = lo + 1.0
                bounds[col] = (lo, hi)

            sim_values: dict[str, float] = {}
            slider_cols = st.columns(2)
            for i, (col, (label, desc)) in enumerate(_FEATURE_LABELS.items()):
                lo, hi = bounds[col]
                default = float(np.clip(result[col], lo, hi))
                # Key includes customer ID so sliders reset when customer changes
                with slider_cols[i % 2]:
                    sim_values[col] = st.slider(
                        label,
                        min_value=lo,
                        max_value=hi,
                        value=default,
                        key=f"sim_{cid_input}_{col}",
                        help=desc,
                    )

            sim_clv = simulate_prediction(sim_values, pipeline)
            delta_clv = sim_clv - pred_clv

            sim_c1, sim_c2, sim_c3 = st.columns(3)
            sim_c1.metric(
                "Simulated CLV",
                f"£{sim_clv:,.2f}",
                delta=f"{'+'if delta_clv >= 0 else ''}{delta_clv:,.2f} vs actual",
                delta_color="normal" if delta_clv >= 0 else "inverse",
            )
            sim_c2.metric(
                "Actual Predicted CLV",
                f"£{pred_clv:,.2f}",
            )
            sim_c3.metric(
                "Change",
                f"{(delta_clv / max(abs(pred_clv), 1)) * 100:+.1f}%",
            )

            st.divider()

            # ----------------------------------------------------------------
            # SHAP Waterfall
            # ----------------------------------------------------------------
            st.subheader("Why this prediction? (SHAP Waterfall)")
            st.caption(
                "Red bars push the prediction **higher** than the base value. "
                "Blue bars push it **lower**."
            )
            with st.spinner("Computing SHAP values…"):
                shap_result = get_shap_explanation(
                    cid_input.strip(), features, pipeline
                )
            if shap_result is not None:
                shap_vals, _ = shap_result
                fig_shap, _ = plt.subplots(figsize=(9, 4))
                shap.plots.waterfall(shap_vals[0], show=False, max_display=10)
                plt.tight_layout()
                st.pyplot(fig_shap, use_container_width=True)
                plt.close()
    else:
        st.info(
            "Enter a Customer ID above or click **🎲 Random Customer** to get started."
        )

# ===========================================================================
# Tab 2 — Segment Explorer + CLV × Churn Risk Matrix
# ===========================================================================

with tab2:
    st.subheader("Segment Explorer")
    st.markdown(
        "Explore all customers across **RFM segments** — groups defined by how recently, "
        "how often, and how much each customer buys. Use the filters to zoom into a country, "
        "CLV range, or specific segments. The **CLV × Churn Risk Matrix** below the charts "
        "maps every customer to one of four business actions."
    )

    with st.expander("📋 RFM Segment Definitions"):
        st.markdown(
            """
Segments are assigned using quartile-based Recency (R) and Frequency (F) scores (1 = worst, 4 = best).

| Segment | R Score | F Score | Who they are | Suggested action |
|---------|---------|---------|--------------|-----------------|
| 👑 **Champions** | 4 | ≥ 3 | Bought recently and often — your best customers | Reward with loyalty perks, early access |
| 💙 **Loyal** | ≥ 3 | ≥ 3 | Frequent buyers, still active | Upsell premium products |
| ⚠️ **At Risk** | ≤ 2 | ≥ 3 | Used to buy often but haven't recently | Win-back campaign, personalised offer |
| 🌱 **Promising** | ≥ 3 | ≤ 2 | Recent first-timers with potential | Onboarding sequence, second-purchase incentive |
| 😴 **Lost** | ≤ 2 | ≤ 2 | Low recency and frequency — likely churned | Re-engagement email or deprioritise |
| ⚪ **Others** | — | — | Customers that don't clearly fit above buckets | Monitor |

> **R score** — higher = bought more recently. **F score** — higher = more unique invoices.
            """
        )

    # Filters
    f1, f2, f3 = st.columns([2, 2, 3])
    with f1:
        countries = ["All"] + sorted(
            all_preds["primary_country"].dropna().unique().tolist()
        )
        sel_country = st.selectbox("Country", countries, index=0)
    with f2:
        all_segs = sorted(all_preds["rfm_segment"].unique().tolist())
        sel_segs = st.multiselect("RFM Segments", all_segs, default=all_segs)
    with f3:
        clv_min = float(all_preds["predicted_clv"].min())
        clv_max = float(all_preds["predicted_clv"].max())
        clv_range = st.slider(
            "Predicted CLV range (£)",
            min_value=clv_min,
            max_value=clv_max,
            value=(clv_min, clv_max),
            format="£%.0f",
        )

    mask = (
        all_preds["predicted_clv"].between(*clv_range)
        & all_preds["rfm_segment"].isin(sel_segs)
    )
    if sel_country != "All":
        mask &= all_preds["primary_country"] == sel_country
    filtered = all_preds[mask]
    st.caption(f"Showing **{len(filtered):,}** of {len(all_preds):,} customers")

    # Full-width scatter — legend floats above the chart
    fig_scatter = px.scatter(
        filtered.reset_index(),
        x="frequency",
        y="predicted_clv",
        color="rfm_segment",
        color_discrete_map=_SEGMENT_COLORS,
        hover_data={
            "Customer ID": True,
            "monetary_total": ":.2f",
            "recency": True,
            "predicted_clv": ":.2f",
            "rfm_segment": False,
        },
        labels={
            "frequency": "Frequency (# invoices)",
            "predicted_clv": "Predicted CLV (£)",
            "rfm_segment": "Segment",
            "monetary_total": "Total Spend (£)",
        },
        title="Frequency vs. Predicted CLV by RFM Segment",
        opacity=0.75,
    )
    fig_scatter.update_traces(marker=dict(size=7))
    fig_scatter.update_layout(
        height=440,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            title="",
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Donut (left) + Avg CLV per segment bar (right) — each given equal space
    seg_counts = filtered["rfm_segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]

    col_donut, col_bar = st.columns(2)

    with col_donut:
        fig_donut = px.pie(
            seg_counts,
            names="Segment",
            values="Count",
            color="Segment",
            color_discrete_map=_SEGMENT_COLORS,
            hole=0.52,
            title="Customer Count by Segment",
        )
        fig_donut.update_traces(
            textposition="inside",
            textinfo="percent",
            textfont_size=13,
            insidetextorientation="radial",
        )
        fig_donut.update_layout(
            height=360,
            showlegend=True,
            legend=dict(
                orientation="v",
                x=1.0,
                xanchor="left",
                y=0.5,
                yanchor="middle",
                font=dict(size=12),
            ),
            margin=dict(l=10, r=130, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_bar:
        seg_clv = (
            filtered.groupby("rfm_segment")["predicted_clv"]
            .mean()
            .round(2)
            .reset_index()
            .rename(columns={"rfm_segment": "Segment", "predicted_clv": "Avg CLV (£)"})
            .sort_values("Avg CLV (£)")
        )
        fig_bar = px.bar(
            seg_clv,
            x="Avg CLV (£)",
            y="Segment",
            color="Segment",
            color_discrete_map=_SEGMENT_COLORS,
            orientation="h",
            title="Average Predicted CLV by Segment",
            text="Avg CLV (£)",
        )
        fig_bar.update_traces(
            texttemplate="£%{text:,.0f}",
            textposition="outside",
        )
        fig_bar.update_layout(
            height=360,
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=60, t=50, b=10),
            xaxis_title="Avg Predicted CLV (£)",
            yaxis_title="",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Segment summary table
    st.subheader("Segment Summary")
    summary = (
        filtered.groupby("rfm_segment")
        .agg(
            Customers=("predicted_clv", "count"),
            **{"Avg CLV (£)": ("predicted_clv", "mean")},
            **{"Median CLV (£)": ("predicted_clv", "median")},
            **{"Avg Frequency": ("frequency", "mean")},
            **{"Avg Recency (days)": ("recency", "mean")},
            **{"Avg Total Spend (£)": ("monetary_total", "mean")},
        )
        .round(2)
        .sort_values("Avg CLV (£)", ascending=False)
        .reset_index()
        .rename(columns={"rfm_segment": "Segment"})
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # CSV download
    csv_cols = [
        "Customer ID", "rfm_segment", "predicted_clv", "frequency",
        "recency", "monetary_total", "monetary_mean", "primary_country",
    ]
    if "churn_proba" in filtered.columns:
        csv_cols.append("churn_proba")
    csv = (
        filtered.reset_index()[csv_cols]
        .rename(columns={"rfm_segment": "segment", "predicted_clv": "predicted_clv_gbp"})
        .to_csv(index=False)
    )
    st.download_button(
        "⬇️ Download filtered customers as CSV",
        data=csv,
        file_name="clv_customers_filtered.csv",
        mime="text/csv",
    )

    # ----------------------------------------------------------------
    # CLV × Churn Risk Matrix
    # ----------------------------------------------------------------
    if "churn_proba" in filtered.columns and filtered["churn_proba"].notna().any():
        st.divider()
        st.subheader("CLV × Churn Risk Matrix")
        st.caption(
            "Each customer is placed in one of four action quadrants based on their "
            "predicted CLV (vertical) and churn probability (horizontal). "
            "Use this to prioritise which customers to act on first."
        )

        matrix_df = filtered.dropna(subset=["churn_proba"]).reset_index()
        clv_mid = float(matrix_df["predicted_clv"].median())
        churn_mid = 0.5  # natural midpoint for probability

        def _quadrant(clv: float, cp: float) -> str:
            if clv >= clv_mid and cp < churn_mid:
                return "👑 VIP — Reward"
            elif clv >= clv_mid and cp >= churn_mid:
                return "🚨 Save Now — Urgent"
            elif clv < clv_mid and cp < churn_mid:
                return "🌱 Nurture — Grow"
            else:
                return "💤 Let Go — Low Priority"

        matrix_df["quadrant"] = [
            _quadrant(c, p)
            for c, p in zip(matrix_df["predicted_clv"], matrix_df["churn_proba"])
        ]

        quad_colors = {
            "👑 VIP — Reward": "#2ecc71",
            "🚨 Save Now — Urgent": "#e74c3c",
            "🌱 Nurture — Grow": "#3498db",
            "💤 Let Go — Low Priority": "#95a5a6",
        }

        fig_matrix = px.scatter(
            matrix_df,
            x="churn_proba",
            y="predicted_clv",
            color="quadrant",
            color_discrete_map=quad_colors,
            hover_data={
                "Customer ID": True,
                "predicted_clv": ":.2f",
                "churn_proba": ":.2f",
                "rfm_segment": True,
                "quadrant": False,
            },
            labels={
                "churn_proba": "Churn Probability →",
                "predicted_clv": "Predicted CLV (£) ↑",
                "quadrant": "Action",
            },
            opacity=0.7,
            title="CLV × Churn Risk — Action Matrix",
        )
        fig_matrix.add_vline(
            x=churn_mid,
            line_dash="dash",
            line_color="white",
            line_width=1,
            annotation_text="50% churn threshold",
            annotation_position="top left",
        )
        fig_matrix.add_hline(
            y=clv_mid,
            line_dash="dash",
            line_color="white",
            line_width=1,
            annotation_text=f"Median CLV £{clv_mid:,.0f}",
            annotation_position="bottom right",
        )
        fig_matrix.update_traces(marker=dict(size=6))
        fig_matrix.update_layout(
            height=460,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                title="Action",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
            xaxis=dict(range=[-0.02, 1.02], tickformat=".0%"),
        )
        st.plotly_chart(fig_matrix, use_container_width=True)

        # Quadrant summary
        quad_summary = (
            matrix_df.groupby("quadrant")
            .agg(
                Customers=("Customer ID", "count"),
                **{"Avg CLV (£)": ("predicted_clv", "mean")},
                **{"Avg Churn Prob": ("churn_proba", "mean")},
            )
            .round(3)
            .reset_index()
            .rename(columns={"quadrant": "Action Quadrant"})
            .sort_values("Avg CLV (£)", ascending=False)
        )
        st.dataframe(quad_summary, use_container_width=True, hide_index=True)
    else:
        st.info(
            "Re-run `python src/train.py` to generate churn predictions "
            "and unlock the CLV × Churn Risk Matrix."
        )

# ===========================================================================
# Tab 3 — Model Performance
# ===========================================================================

with tab3:
    st.subheader("CLV Model Performance")
    st.markdown(
        "This tab evaluates how accurately the XGBoost model predicts future revenue, "
        "how well the churn classifier identifies at-risk customers, and how the two models "
        "compare against the probabilistic BG/NBD baseline."
    )

    with st.expander("📐 What do these metrics mean?"):
        st.markdown(
            """
**CLV Regression Metrics**

| Metric | Ideal | Interpretation |
|--------|-------|---------------|
| **MAE** (Mean Absolute Error) | As low as possible | On average, the model's prediction is off by this many £. |
| **RMSE** (Root Mean Squared Error) | As low as possible | Like MAE but penalises large errors more heavily. |
| **R²** (R-squared) | Closer to 1.0 | The proportion of variance in CLV explained by the model. An R² of 0.80 means the model explains 80% of the variation. |
| **MAPE** (Mean Absolute % Error) | As low as possible | Average percentage error — useful for comparing across different revenue scales. |
| **Pearson r** | Closer to 1.0 | Linear correlation between actual and predicted CLV. |

**Churn Classification Metrics**

| Metric | Ideal | Interpretation |
|--------|-------|---------------|
| **AUC-ROC** | Closer to 1.0 | How well the model separates churners from returners. 0.5 = random guessing; 1.0 = perfect. |
| **Precision** | Higher is better | Of customers predicted to return, the fraction who actually did. |
| **Recall** | Higher is better | Of customers who actually returned, the fraction the model correctly flagged. |
| **F1** | Higher is better | The harmonic mean of Precision and Recall — useful when classes are imbalanced. |
            """
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAE (£)", f"{metrics['MAE']:,.2f}", help="Mean Absolute Error")
    c2.metric("RMSE (£)", f"{metrics['RMSE']:,.2f}", help="Root Mean Squared Error")
    c3.metric("R²", f"{metrics['R2']:.4f}", help="1.0 = perfect fit")
    c4.metric("MAPE (%)", f"{metrics['MAPE']:.2f}", help="Mean Absolute Percentage Error")
    c5.metric("Pearson r", f"{metrics['Pearson_r']:.4f}", help="Actual vs. predicted correlation")

    st.divider()

    col_l, col_r = st.columns(2)

    # Interactive actual vs. predicted
    with col_l:
        st.subheader("Actual vs. Predicted CLV")
        tp_path = OUTPUTS_DIR / "test_predictions.parquet"
        if tp_path.exists():
            tp = pd.read_parquet(tp_path)
            lims = [
                min(float(tp["y_test"].min()), float(tp["y_pred"].min())),
                max(float(tp["y_test"].max()), float(tp["y_pred"].max())),
            ]
            fig_avp = go.Figure()
            fig_avp.add_trace(
                go.Scatter(
                    x=tp["y_test"],
                    y=tp["y_pred"],
                    mode="markers",
                    marker=dict(color="steelblue", opacity=0.45, size=5),
                    name="Customers",
                    hovertemplate="Actual: £%{x:,.0f}<br>Predicted: £%{y:,.0f}<extra></extra>",
                )
            )
            fig_avp.add_trace(
                go.Scatter(
                    x=lims, y=lims,
                    mode="lines",
                    line=dict(color="red", dash="dash", width=1),
                    name="Perfect prediction",
                )
            )
            fig_avp.update_layout(
                xaxis_title="Actual Revenue (£)",
                yaxis_title="Predicted Revenue (£)",
                height=360,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_avp, use_container_width=True)
        else:
            st.info("Run `python src/train.py` to generate test predictions.")

    # Residuals histogram
    with col_r:
        st.subheader("Residuals Distribution")
        if tp_path.exists():
            tp["residual"] = tp["y_pred"] - tp["y_test"]
            fig_resid = px.histogram(
                tp, x="residual", nbins=60,
                color_discrete_sequence=["steelblue"],
                labels={"residual": "Residual (Predicted − Actual) £"},
            )
            fig_resid.add_vline(x=0, line_dash="dash", line_color="red",
                                annotation_text="Zero error")
            fig_resid.update_layout(
                height=360,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig_resid, use_container_width=True)

    st.divider()

    # SHAP plots toggle
    st.subheader("SHAP Global Feature Importance")
    plot_choice = st.radio(
        "Plot type", ["Bar (mean |SHAP|)", "Beeswarm"], horizontal=True
    )
    img_path = (
        OUTPUTS_DIR / "shap_summary.png"
        if plot_choice == "Bar (mean |SHAP|)"
        else OUTPUTS_DIR / "shap_beeswarm.png"
    )
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)
    else:
        st.info("Run `python src/train.py` to generate SHAP plots.")

    # CLV distribution
    st.divider()
    st.subheader("Predicted CLV Distribution by Segment")
    fig_dist = px.histogram(
        all_preds, x="predicted_clv", color="rfm_segment",
        color_discrete_map=_SEGMENT_COLORS, nbins=80, opacity=0.75,
        barmode="overlay",
        labels={"predicted_clv": "Predicted CLV (£)", "rfm_segment": "Segment"},
    )
    fig_dist.add_vline(
        x=_threshold, line_dash="dash", line_color="white",
        annotation_text=f"High-value threshold £{_threshold:,.0f}",
        annotation_position="top right",
    )
    fig_dist.update_layout(
        height=300,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # ----------------------------------------------------------------
    # Churn Model Performance
    # ----------------------------------------------------------------
    if churn_metrics:
        st.divider()
        st.subheader("Churn Model Performance")
        st.caption(
            "Binary classifier: predicts whether a customer will make any purchase "
            "in the 3-month prediction window (Jan–Mar 2011)."
        )

        ch1, ch2, ch3, ch4, ch5 = st.columns(5)
        ch1.metric("AUC-ROC", f"{churn_metrics.get('AUC_ROC', 0):.4f}",
                   help="Area under the ROC curve. 1.0 = perfect, 0.5 = random.")
        ch2.metric("Accuracy", f"{churn_metrics.get('Accuracy', 0):.4f}")
        ch3.metric("Precision (returned)",
                   f"{churn_metrics.get('Precision_returned', 0):.4f}",
                   help="Of customers predicted to return, how many actually did.")
        ch4.metric("Recall (returned)",
                   f"{churn_metrics.get('Recall_returned', 0):.4f}",
                   help="Of customers who actually returned, how many were caught.")
        ch5.metric("F1 (returned)",
                   f"{churn_metrics.get('F1_returned', 0):.4f}")

        roc_path = OUTPUTS_DIR / "churn_roc_curve.parquet"
        if roc_path.exists():
            roc_df = pd.read_parquet(roc_path)
            fig_roc = go.Figure()
            fig_roc.add_trace(
                go.Scatter(
                    x=roc_df["fpr"], y=roc_df["tpr"],
                    mode="lines",
                    name=f"Churn model (AUC = {churn_metrics.get('AUC_ROC', 0):.3f})",
                    line=dict(color="#e74c3c", width=2),
                )
            )
            fig_roc.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode="lines",
                    line=dict(color="gray", dash="dash", width=1),
                    name="Random classifier",
                )
            )
            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                title="ROC Curve — Churn Classifier",
                height=380,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_roc, use_container_width=True)
    else:
        st.info(
            "Churn model metrics not found. Re-run `python src/train.py` to train "
            "the churn classifier and unlock this section."
        )

    # ----------------------------------------------------------------
    # BG/NBD vs XGBoost Model Comparison
    # ----------------------------------------------------------------
    if bgn_preds is not None and "bgnbd_clv" in all_preds.columns:
        st.divider()
        st.subheader("Model Comparison: XGBoost vs BG/NBD + Gamma-Gamma")
        st.markdown(
            """
**XGBoost** is a supervised machine learning model — it learns patterns directly from the labelled
training data (observed CLV) and optimises for prediction accuracy.

**BG/NBD + Gamma-Gamma** is an unsupervised probabilistic model from academic marketing research.
It makes no use of the actual future revenue labels; instead it models the *process* by which
customers buy and drop out, independently estimating how many transactions to expect and at what value.

When both models produce similar predictions for a customer, that agreement increases confidence.
When they disagree significantly, that customer is worth investigating further.
A high Pearson r between the two models indicates they capture consistent signal from the data.
            """
        )

        compare_df = all_preds.dropna(subset=["bgnbd_clv", "predicted_clv"])

        cmp_l, cmp_r = st.columns(2)

        with cmp_l:
            # Scatter: XGBoost vs BG/NBD CLV
            fig_cmp = px.scatter(
                compare_df.reset_index(),
                x="predicted_clv",
                y="bgnbd_clv",
                color="rfm_segment",
                color_discrete_map=_SEGMENT_COLORS,
                opacity=0.55,
                labels={
                    "predicted_clv": "XGBoost Predicted CLV (£)",
                    "bgnbd_clv": "BG/NBD CLV (£)",
                    "rfm_segment": "Segment",
                },
                title="XGBoost vs BG/NBD CLV — per customer",
                hover_data={"Customer ID": True, "predicted_clv": ":.0f", "bgnbd_clv": ":.0f"},
            )
            # Perfect agreement line
            lim = float(compare_df[["predicted_clv", "bgnbd_clv"]].max().max())
            fig_cmp.add_trace(
                go.Scatter(
                    x=[0, lim], y=[0, lim],
                    mode="lines",
                    line=dict(color="white", dash="dash", width=1),
                    name="Perfect agreement",
                    showlegend=True,
                )
            )
            corr = compare_df["predicted_clv"].corr(compare_df["bgnbd_clv"])
            fig_cmp.update_layout(
                height=400,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                annotations=[
                    dict(
                        text=f"Pearson r = {corr:.3f}",
                        xref="paper", yref="paper",
                        x=0.02, y=0.97,
                        showarrow=False,
                        font=dict(size=13, color="white"),
                        bgcolor="rgba(0,0,0,0.4)",
                    )
                ],
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

        with cmp_r:
            # Overlaid CLV distributions
            fig_overlay = go.Figure()
            fig_overlay.add_trace(
                go.Histogram(
                    x=compare_df["predicted_clv"],
                    nbinsx=60,
                    name="XGBoost",
                    marker_color="steelblue",
                    opacity=0.65,
                )
            )
            fig_overlay.add_trace(
                go.Histogram(
                    x=compare_df["bgnbd_clv"],
                    nbinsx=60,
                    name="BG/NBD",
                    marker_color="#e67e22",
                    opacity=0.65,
                )
            )
            fig_overlay.update_layout(
                barmode="overlay",
                title="CLV Distribution: XGBoost vs BG/NBD",
                xaxis_title="Predicted CLV (£)",
                yaxis_title="Customers",
                height=400,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_overlay, use_container_width=True)

        # Summary comparison table
        cmp_stats = pd.DataFrame(
            {
                "Model": ["XGBoost", "BG/NBD + Gamma-Gamma"],
                "Mean CLV (£)": [
                    round(compare_df["predicted_clv"].mean(), 2),
                    round(compare_df["bgnbd_clv"].mean(), 2),
                ],
                "Median CLV (£)": [
                    round(compare_df["predicted_clv"].median(), 2),
                    round(compare_df["bgnbd_clv"].median(), 2),
                ],
                "Std Dev (£)": [
                    round(compare_df["predicted_clv"].std(), 2),
                    round(compare_df["bgnbd_clv"].std(), 2),
                ],
            }
        )
        st.dataframe(cmp_stats, use_container_width=True, hide_index=True)
    else:
        st.info(
            "BG/NBD predictions not found. Re-run `python src/train.py` to generate "
            "them and unlock the model comparison section."
        )
