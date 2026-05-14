import streamlit as st
import numpy as np
import pickle

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation App",
    page_icon="🎯",
    layout="centered"
)

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

# ── Segment labels ────────────────────────────────────────────
SEGMENT_LABELS = {
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

# ── Header ────────────────────────────────────────────────────
st.title("🎯 Customer Segmentation App")
st.markdown("#### Predict which customer segment a customer belongs to using RFM Analysis")
st.markdown(
    "Built by **[Bindhu Saahithi](https://github.com/bindhusaahithi)** · "
    "K-Means Clustering + Random Forest · "
    "[View on GitHub](https://github.com/bindhusaahithi/Customer-Segmentation-Customer-Lifetime-Value-Prediction)"
)
st.markdown("---")

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
    🔗 [GitHub](https://github.com/bindhusaahithi) | [LinkedIn](https://www.linkedin.com/in/bindhu-saahithi-naralashetty-yogendranath/) | [Kaggle](https://www.kaggle.com/bindhusaahithi)
    """)

# ── Sample data buttons ───────────────────────────────────────
st.markdown("### 🎲 Try a Sample Customer")
col_s1, col_s2, col_s3, col_s4 = st.columns(4)

sample_recency   = st.session_state.get("recency", 30)
sample_frequency = st.session_state.get("frequency", 10)
sample_monetary  = st.session_state.get("monetary", 500.0)

with col_s1:
    if st.button("💎 VIP"):
        st.session_state["recency"]   = 5
        st.session_state["frequency"] = 80
        st.session_state["monetary"]  = 8000.0
        st.rerun()

with col_s2:
    if st.button("😴 At-Risk"):
        st.session_state["recency"]   = 300
        st.session_state["frequency"] = 3
        st.session_state["monetary"]  = 200.0
        st.rerun()

with col_s3:
    if st.button("🌱 New"):
        st.session_state["recency"]   = 20
        st.session_state["frequency"] = 2
        st.session_state["monetary"]  = 150.0
        st.rerun()

with col_s4:
    if st.button("📦 Regular"):
        st.session_state["recency"]   = 45
        st.session_state["frequency"] = 15
        st.session_state["monetary"]  = 900.0
        st.rerun()

st.markdown("---")

# ── Input form ────────────────────────────────────────────────
st.markdown("### 📋 Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    recency = st.slider(
        "📅 Recency (days since last purchase)",
        min_value=1, max_value=365,
        value=st.session_state.get("recency", 30),
        help="Lower = more recent purchase"
    )

with col2:
    frequency = st.slider(
        "🔁 Frequency (number of purchases)",
        min_value=1, max_value=200,
        value=st.session_state.get("frequency", 10),
        help="Higher = buys more often"
    )

with col3:
    monetary = st.number_input(
        "💰 Monetary (total spend £)",
        min_value=0.0, max_value=50000.0,
        value=float(st.session_state.get("monetary", 500.0)),
        step=50.0,
        help="Total amount spent by the customer"
    )

st.markdown("---")

# ── Predict button ────────────────────────────────────────────
if st.button("🔍 Predict Customer Segment", use_container_width=True):

    # Prepare inputs
    input_data   = np.array([[recency, frequency, monetary]])
    input_scaled = scaler.transform(input_data)
    input_rf     = np.array([[recency, frequency]])

    # Predictions
    cluster         = kmeans.predict(input_scaled)[0]
    predicted_spend = rf_model.predict(input_rf)[0]

    label, color, description = SEGMENT_LABELS.get(
        cluster, ("Unknown", "#636e72", "")
    )

    # ── Result card ───────────────────────────────────────────
    st.markdown("### 📊 Prediction Results")
    st.markdown(
        f"""
        <div style="
            background-color:{color}22;
            border-left:5px solid {color};
            padding:20px;
            border-radius:10px;
            margin:10px 0;
        ">
            <h2 style="color:{color};margin:0;">{label}</h2>
            <p style="color:#636e72;margin-top:8px;">{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Cluster",         f"#{cluster}")
    c2.metric("Predicted Spend", f"£{predicted_spend:,.0f}")
    c3.metric("RFM Input",       f"R:{recency} F:{frequency}")

    # Recommendations
    st.markdown("### 💡 Recommended Actions")
    for rec in RECOMMENDATIONS.get(cluster, []):
        st.markdown(f"- {rec}")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;color:#636e72;font-size:13px;">
        Built by
        <a href="https://github.com/bindhusaahithi" target="_blank">Bindhu Saahithi</a>
        ·
        <a href="https://www.linkedin.com/in/bindhu-saahithi-naralashetty-yogendranath/" target="_blank">LinkedIn</a>
        ·
        <a href="https://www.kaggle.com/bindhusaahithi" target="_blank">Kaggle</a>
        ·
        <a href="https://github.com/bindhusaahithi/Customer-Segmentation-Customer-Lifetime-Value-Prediction" target="_blank">View on GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
