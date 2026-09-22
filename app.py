"""
app.py
------
Interactive Streamlit demo: enter transaction details, get a fraud
probability score AND a SHAP-based explanation of why the model made
that call - this is the piece that turns "I trained a model" into
"I built a decision-support tool a non-technical fraud analyst could use."

Run with: streamlit run app.py
"""

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="Fraud Detection Demo", layout="centered")

FEATURE_COLS = [
    "transaction_amount", "hour_of_day", "distance_from_home_km",
    "distance_from_last_txn_km", "txns_last_24h", "avg_txn_amount_last_30d",
    "is_online", "merchant_risk_score", "card_present",
    "amount_to_avg_ratio", "is_night_txn", "online_and_not_present", "high_velocity",
]


@st.cache_resource
def load_model():
    return joblib.load("models/xgb_model.joblib")


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


model = load_model()
explainer = load_explainer(model)

st.title("💳 Explainable Fraud Detection")
st.caption("Hybrid XGBoost + Isolation Forest model, explained with SHAP")

st.subheader("Enter transaction details")

col1, col2 = st.columns(2)
with col1:
    amount = st.number_input("Transaction amount ($)", min_value=1.0, value=150.0)
    hour = st.slider("Hour of day", 0, 23, 14)
    dist_home = st.number_input("Distance from home (km)", min_value=0.0, value=10.0)
    dist_last = st.number_input("Distance from last transaction (km)", min_value=0.0, value=5.0)
    txns_24h = st.number_input("Transactions in last 24h", min_value=0, value=2, step=1)
    avg_30d = st.number_input("Average transaction amount, last 30 days ($)", min_value=1.0, value=100.0)

with col2:
    is_online = st.checkbox("Online transaction?", value=False)
    card_present = st.checkbox("Card physically present?", value=True)
    merchant_risk = st.slider("Merchant risk score (0=safe, 1=risky)", 0.0, 1.0, 0.2)

if st.button("Score this transaction", type="primary"):
    amount_to_avg_ratio = amount / max(avg_30d, 1)
    is_night = 1 if hour <= 5 else 0
    online_not_present = 1 if (is_online and not card_present) else 0
    # For the demo, "high velocity" uses the same 90th-percentile-style heuristic
    # as training (approximated here with a fixed threshold for simplicity).
    high_velocity = 1 if txns_24h > 6 else 0

    row = pd.DataFrame([{
        "transaction_amount": amount,
        "hour_of_day": hour,
        "distance_from_home_km": dist_home,
        "distance_from_last_txn_km": dist_last,
        "txns_last_24h": txns_24h,
        "avg_txn_amount_last_30d": avg_30d,
        "is_online": int(is_online),
        "merchant_risk_score": merchant_risk,
        "card_present": int(card_present),
        "amount_to_avg_ratio": amount_to_avg_ratio,
        "is_night_txn": is_night,
        "online_and_not_present": online_not_present,
        "high_velocity": high_velocity,
    }])[FEATURE_COLS]

    prob = model.predict_proba(row)[0, 1]

    st.divider()
    if prob >= 0.5:
        st.error(f"⚠️ FLAGGED AS FRAUD — probability: {prob:.1%}")
    else:
        st.success(f"✅ Looks legitimate — fraud probability: {prob:.1%}")

    st.subheader("Why did the model decide this?")
    shap_values = explainer(row)

    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

    st.caption(
        "Red bars push the prediction toward FRAUD, blue bars push it toward LEGIT. "
        "Bar size = how much that feature influenced this specific decision."
    )
