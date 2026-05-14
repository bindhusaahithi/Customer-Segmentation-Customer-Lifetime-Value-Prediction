import streamlit as st
import numpy as np
import pickle

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation App",
    page_icon="🎯",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide default streamlit header */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #6C5CE7 0%, #0984E3 50%, #00B894 100%);
        padding: 40px 30px;
        border-radius: 16px;
        margin-bottom: 24px;
        text-align: center;
    }
    .hero-title {
        color: white;
        font-size: 2.4em;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: rgba(255,255,255,0.85);
        font-size: 1.1em;
        margin-top: 8px;
    }
    .hero-links a {
        color: rgba(255,255,255,0.9);
        text-decoration: none;
        margin: 0 10px;
        font-size: 0.95em;
        border-bottom: 1px solid rgba(255,255,255,0.4);
    }

    /* Stats bar */
    .stats-bar {
        display: flex;
        justify-content: space-around;
        background: #1a1a2e;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 24px;
        border: 1px solid #2d2d44;
    }
    .stat-item {
        text-align: center;
    }
    .stat-number {
        font-size: 1.8em;
        font-weight: 800;
        color: #6C5CE7;
        display: block;
    }
    .stat-label {
        font-size: 0.8em;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Sample buttons */
    .sample-label {
        font-size: 1.1em;
        font-weight: 600;
        margin-bottom: 8px;
        color: #ddd;
    }

    /* Segment result card */
    .result-card {
        border-radius: 14px;
        padding: 24px;
        margin: 16px 0;
    }
    .result-title {
        font-size: 1.8em;
        font-weight: 800;
        margin: 0;
    }
    .result-desc {
        color: #aaa;
        margin-top: 8px;
        font-size: 1em;
    }

    /* Section headers */
    .section-header {
        font-size: 1.2em;
        font-weight: 700;
        color: #ddd;
        margin: 24px 0 12px 0;
        padding-bottom: 6px;
        border-bottom: 2px solid #2d2d44;
    }

    /* Predict button */
    .stButton > button {
        background: linear-gradient(135deg, #6C5CE7, #0984E3) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-size: 1.1em !important;
        font-weight: 600 !important;
        padding: 14px !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover {
        opacity: 0.9 !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.85em;
        padding: 20px 0 10px 0;
    }
    .footer a {
        color: #6C5CE7;
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("rf_model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    return kmeans, scaler, rf_model

kmeans, scaler, rf_model = load_models()

# ── Segment config ────────────────────────────────────────────
SEGMENTS = {
    0: ("💎 High Value Customer",     "#6C5CE7", "Purchases frequently and spends a lot. Top priority for retention."),
    1: ("😴 At-Risk Customer",        "#E17055", "Has not purchased recently. Needs re-engagement campaigns."),
    2: ("🌱 New / Potential Customer", "#00B894", "Low frequency but shows promise. Nurture with targeted offers."),
    3: ("📦 Regular Customer",        "#0984E3", "Moderate activity. Consistent but not a top spender."),
}

RECOMMENDATIONS = {
    0: ["🎁 Offer exclusive loyalty rewards", "📧 Send VIP early access emails", "🏆 Create a premium membership tier"],
    1: ["📬 Send win-back email campaign", "💸 Offer a special discount", "📱 Push notification with personalized offer"],
    2: ["👋 Send welcome series emails", "🛍️ Recommend popular products", "⭐ Offer first purchase incentive"],
    3: ["📦 Upsell complementary products", "🔔 Set up restock alerts", "📊 Share personalized recommendations"],
}

# ── Hero Banner ───────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">🎯 Customer Segmentation App</p>
    <p class="hero-subtitle">Predict customer segments using RFM Analysis · K-Means + Random Forest</p>
    <div class="hero-links" style="margin-top:14px;">
        <a href="https://github.com/bindhusaahithi" target="_blank">GitHub</a>
        <a href="https://www.linkedin.com/in/bindhu-saahithi-naralashetty-yogendranath/" target="_blank">LinkedIn</a>
        <a href="https://www.kaggle.com/bindhusaahithi" target="_blank">Kaggle</a>
        <a href="https://github.com/bindhusaahithi/Customer-Segmentation-Customer-Lifetime-Value-Prediction" target="_blank">View Code</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Stats Bar ─────────────────────────────────────────────────
st.markdown("""
<div class="stats-bar">
    <div class="stat-item">
        <span class="stat-number">500K+</span>
        <span class="stat-label">Transactions</span>
    </div>
    <div class="stat-item">
        <span class="stat-number">4,338</span>
        <span class="stat-label">Customers</span>
    </div>
    <div class="stat-item">
        <span class="stat-number">4</span>
        <span class="stat-label">Segments</span>
    </div>
    <div class="stat-item">
        <span class="stat-number">0.45</span>
        <span class="stat-label">R² Score</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── About section ─────────────────────────────────────────────
with st.expander("ℹ️ What is RFM Analysis?"):
    st.markdown("""
    **RFM** stands for:
    - **Recency** — How recently did the customer purchase?
    - **Frequency** — How often do they purchase?
    - **Monetary** — How much do they spend in total?

    This app uses **K-Means Clustering** to segment customers into 4 groups
    and **Random Forest** to predict their future spend.

    📊 Built with the Online Retail dataset (UK-based e-commerce, 500K+ transactions)
    """)

# ── Sample buttons ────────────────────────────────────────────
st.markdown('<p class="section-header">🎲 Try a Sample Customer</p>', unsafe_allow_html=True)

col_s1, col_s2, col_s3, col_s4 = st.columns(4)

with col_s1:
    if st.button("💎 VIP", use_container_width=True):
        st.session_state.update({"recency": 5, "frequency": 80, "monetary": 8000.0})
        st.rerun()
with col_s2:
    if st.button("😴 At-Risk", use_container_width=True):
        st.session_state.update({"recency": 300, "frequency": 3, "monetary": 200.0})
        st.rerun()
with col_s3:
    if st.button("🌱 New", use_container_width=True):
        st.session_state.update({"recency": 20, "frequency": 2, "monetary": 150.0})
        st.rerun()
with col_s4:
    if st.button("📦 Regular", use_container_width=True):
        st.session_state.update({"recency": 45, "frequency": 15, "monetary": 900.0})
        st.rerun()

# ── Input controls ────────────────────────────────────────────
st.markdown('<p class="section-header">📋 Enter Customer Details</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    recency = st.slider(
        "📅 Recency (days)",
        min_value=1, max_value=365,
        value=st.session_state.get("recency", 30),
        help="Days since last purchase. Lower = more recent."
    )
with col2:
    frequency = st.slider(
        "🔁 Frequency (purchases)",
        min_value=1, max_value=200,
        value=st.session_state.get("frequency", 10),
        help="Total number of purchases made."
    )
with col3:
    monetary = st.number_input(
        "💰 Monetary (£ spend)",
        min_value=0.0, max_value=50000.0,
        value=float(st.session_state.get("monetary", 500.0)),
        step=50.0,
        help="Total amount spent by the customer."
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────────
if st.button("🔍 Predict Customer Segment", use_container_width=True):

    input_data   = np.array([[recency, frequency, monetary]])
    input_scaled = scaler.transform(input_data)
    input_rf     = np.array([[recency, frequency]])

    cluster         = kmeans.predict(input_scaled)[0]
    predicted_spend = rf_model.predict(input_rf)[0]

    label, color, description = SEGMENTS.get(cluster, ("Unknown", "#636e72", ""))

    st.markdown('<p class="section-header">📊 Prediction Results</p>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-card" style="background:{color}18; border-left:5px solid {color};">
        <p class="result-title" style="color:{color};">{label}</p>
        <p class="result-desc">{description}</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Cluster",         f"#{cluster}")
    c2.metric("Predicted Spend", f"£{predicted_spend:,.0f}")
    c3.metric("Recency / Freq",  f"{recency}d / {frequency}x")

    st.markdown('<p class="section-header">💡 Recommended Actions</p>', unsafe_allow_html=True)
    for rec in RECOMMENDATIONS.get(cluster, []):
        st.markdown(f"- {rec}")

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built by <a href="https://github.com/bindhusaahithi">Bindhu Saahithi</a> ·
    <a href="https://www.linkedin.com/in/bindhu-saahithi-naralashetty-yogendranath/">LinkedIn</a> ·
    <a href="https://www.kaggle.com/bindhusaahithi">Kaggle</a> ·
    <a href="https://github.com/bindhusaahithi/Customer-Segmentation-Customer-Lifetime-Value-Prediction">View on GitHub</a>
</div>
""", unsafe_allow_html=True)
