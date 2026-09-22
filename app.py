"""
app.py
------
Interactive Streamlit demo: enter transaction details, get a fraud
probability score AND a SHAP-based explanation of why the model made
that call.

Run with: streamlit run app.py
"""

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="Fraud Risk Console", layout="wide")

FEATURE_COLS = [
    "transaction_amount", "hour_of_day", "distance_from_home_km",
    "distance_from_last_txn_km", "txns_last_24h", "avg_txn_amount_last_30d",
    "is_online", "merchant_risk_score", "card_present",
    "amount_to_avg_ratio", "is_night_txn", "online_and_not_present", "high_velocity",
]

SAFE_COLOR = "#34A897"
RISK_COLOR = "#E85C41"
PANEL_BG = "#161B29"
BORDER = "#262C3D"
TEXT_MUTED = "#838DA3"

# ---------------------------------------------------------------------------
# Styling: fonts + header bar + verdict card + form panel.
# Streamlit's native theme (.streamlit/config.toml) handles buttons, sliders,
# and backgrounds; this CSS layers on typography and the custom components
# native Streamlit widgets can't produce (header bar, verdict card).
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

h1, h2, h3 {{
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.01em;
}}

.block-container {{
    padding-top: 2rem;
    max-width: 1100px;
}}

.console-header {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    border-bottom: 1px solid {BORDER};
    padding-bottom: 1.1rem;
    margin-bottom: 1.6rem;
}}
.console-header h1 {{
    font-size: 1.7rem;
    font-weight: 600;
    margin: 0;
    color: #E7E9F0;
}}
.console-header .subtitle {{
    color: {TEXT_MUTED};
    font-size: 0.92rem;
    margin-top: 0.3rem;
}}
.status-pill {{
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.82rem;
    color: {TEXT_MUTED};
    white-space: nowrap;
}}
.status-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: {SAFE_COLOR};
    box-shadow: 0 0 6px {SAFE_COLOR};
}}

.panel-label {{
    font-size: 0.78rem;
    color: {TEXT_MUTED};
    font-weight: 500;
    margin-bottom: 0.6rem;
    border-bottom: 1px solid {BORDER};
    padding-bottom: 0.5rem;
}}

.verdict-card {{
    border-radius: 6px;
    padding: 1.3rem 1.5rem;
    border: 1px solid;
    margin-bottom: 1.2rem;
}}
.verdict-card.safe {{
    background: rgba(52, 168, 151, 0.08);
    border-color: {SAFE_COLOR};
}}
.verdict-card.risk {{
    background: rgba(232, 92, 65, 0.08);
    border-color: {RISK_COLOR};
}}
.verdict-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem;
    font-weight: 600;
    margin: 0;
}}
.verdict-title.safe {{ color: {SAFE_COLOR}; }}
.verdict-title.risk {{ color: {RISK_COLOR}; }}
.verdict-score {{
    font-size: 0.88rem;
    color: {TEXT_MUTED};
    margin-top: 0.25rem;
}}

.empty-state {{
    color: {TEXT_MUTED};
    font-size: 0.88rem;
    border: 1px dashed {BORDER};
    border-radius: 6px;
    padding: 1.8rem;
    text-align: center;
    margin-top: 0.5rem;
}}

footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="console-header">
    <div>
        <h1>Fraud Risk Console</h1>
        <div class="subtitle">Score a transaction and see exactly why</div>
    </div>
    <div class="status-pill"><span class="status-dot"></span>Model online — hybrid XGBoost + Isolation Forest</div>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load("models/xgb_model.joblib")


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


model = load_model()
explainer = load_explainer(model)

col_form, col_result = st.columns([1, 1], gap="large")

with col_form:
    st.markdown('<div class="panel-label">Transaction details</div>', unsafe_allow_html=True)

    amount = st.number_input("Transaction amount ($)", min_value=1.0, value=150.0)
    hour = st.slider("Hour of day", 0, 23, 14)
    dist_home = st.number_input("Distance from home (km)", min_value=0.0, value=10.0)
    dist_last = st.number_input("Distance from last transaction (km)", min_value=0.0, value=5.0)
    txns_24h = st.number_input("Transactions in last 24h", min_value=0, value=2, step=1)
    avg_30d = st.number_input("Average transaction amount, last 30 days ($)", min_value=1.0, value=100.0)

    c1, c2 = st.columns(2)
    with c1:
        is_online = st.checkbox("Online transaction?", value=False)
    with c2:
        card_present = st.checkbox("Card physically present?", value=True)

    merchant_risk = st.slider("Merchant risk score (0 = safe, 1 = risky)", 0.0, 1.0, 0.2)

    score_clicked = st.button("Score this transaction", type="primary", use_container_width=True)

with col_result:
    st.markdown('<div class="panel-label">Risk assessment</div>', unsafe_allow_html=True)

    if not score_clicked:
        st.markdown(
            '<div class="empty-state">Enter transaction details and click '
            '"Score this transaction" to see the model\'s decision and explanation.</div>',
            unsafe_allow_html=True,
        )
    else:
        amount_to_avg_ratio = amount / max(avg_30d, 1)
        is_night = 1 if hour <= 5 else 0
        online_not_present = 1 if (is_online and not card_present) else 0
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
        is_fraud = prob >= 0.5

        verdict_class = "risk" if is_fraud else "safe"
        verdict_text = "Flagged as fraud" if is_fraud else "Looks legitimate"

        st.markdown(f"""
        <div class="verdict-card {verdict_class}">
            <div class="verdict-title {verdict_class}">{verdict_text}</div>
            <div class="verdict-score">Fraud probability: {prob:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

        shap_values = explainer(row)

        # Style the SHAP plot to match the console's dark theme instead of
        # matplotlib's default white background.
        plt.rcParams.update({
            "figure.facecolor": PANEL_BG,
            "axes.facecolor": PANEL_BG,
            "savefig.facecolor": PANEL_BG,
            "text.color": "#E7E9F0",
            "axes.labelcolor": "#E7E9F0",
            "xtick.color": TEXT_MUTED,
            "ytick.color": "#E7E9F0",
            "axes.edgecolor": BORDER,
            "font.family": "sans-serif",
        })

        fig, ax = plt.subplots(figsize=(7, 4))
        shap.plots.waterfall(shap_values[0], show=False)
        fig.patch.set_facecolor(PANEL_BG)
        ax.set_facecolor(PANEL_BG)
        st.pyplot(fig, use_container_width=True)

        st.markdown(
            f'<div style="color:{TEXT_MUTED}; font-size:0.82rem; margin-top:0.4rem;">'
            f'Red bars push the prediction toward fraud, teal bars push it toward legitimate. '
            f'Bar size reflects how much that factor influenced this specific decision.</div>',
            unsafe_allow_html=True,
        )
