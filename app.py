import streamlit as st
import pandas as pd
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

# ── Header ────────────────────────────────────────────────────
st.title("🎯 Customer Segmentation App")
st.markdown("#### Predict which customer segment a customer belongs to using RFM Analysis")
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

    📊 Built with the Online Retail dataset (UK-based e-commerce)  
    🔗 [GitHub](https://github.com/bindhusaahithi) | [LinkedIn](https://www.linkedin.com/in/bindhu-saahithi-naralashetty-yogendranath/)
    """)

# ── Input form ────────────────────────────────────────────────
st.markdown("### 📋 Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    recency = st.slider(
        "📅 Recency (days since last purchase)",
        min_value=1,
        max_value=365,
        value=30,
        help="Lower = more recent purchase"
    )

with col2:
    frequency = st.slider(
        "🔁 Frequency (number of purchases)",
        min_value=1,
        max_value=200,
        value=10,
        help="Higher = buys more often"
    )

with col3:
    monetary = st.number_input(
        "💰 Monetary (total spend £)",
        min_value=0.0,
        max_value=50000.0,
        value=500.0,
        step=50.0,
        help="Total amount spent by the customer"
    )

st.markdown("---")

# ── Predict button ────────────────────────────────────────────
if st.button("🔍 Predict Customer Segment", use_container_width=True):

    # Prepare input — 3 features for KMeans
    input_data = np.array([[recency, frequency, monetary]])
    input_scaled = scaler.transform(input_data)

    # Predict segment using KMeans (3 features)
    cluster = kmeans.predict(input_scaled)[0]

    # Predict spend using RandomForest (2 features only)
    input_rf = np.array([[recency, frequency]])
    predicted_spend = rf_model.predict(input_rf)[0]

    # Get label
    label, color, description = SEGMENT_LABELS.get(
        cluster,
        ("Unknown Segment", "#636e72", "")
    )

    # ── Results ───────────────────────────────────────────────
    st.markdown("### 📊 Prediction Results")

    st.markdown(
        f"""
        <div style="
            background-color: {color}22;
            border-left: 5px solid {color};
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        ">
            <h2 style="color: {color}; margin: 0;">{label}</h2>
            <p style="color: #636e72; margin-top: 8px;">{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Metrics
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Cluster", f"#{cluster}")
    col_b.metric("Predicted Spend", f"£{predicted_spend:,.0f}")
    col_c.metric("RFM Score", f"R:{recency} F:{frequency} M:£{monetary:,.0f}")

    # ── Recommendations ───────────────────────────────────────
    st.markdown("### 💡 Recommended Actions")

    recommendations = {
        0: ["🎁 Offer exclusive loyalty rewards", "📧 Send VIP early access emails", "🏆 Create a premium membership tier"],
        1: ["📬 Send win-back email campaign", "💸 Offer a special discount", "📱 Push notification with personalized offer"],
        2: ["👋 Send welcome series emails", "🛍️ Recommend popular products", "⭐ Offer first purchase incentive"],
        3: ["📦 Upsell complementary products", "🔔 Set up restock alerts", "📊 Share personalized recommendations"],
    }

    for rec in recommendations.get(cluster, []):
        st.markdown(f"- {rec}")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #636e72; font-size: 13px;">
        Built by <a href="https://github.com/bindhusaahithi" target="_blank">Bindhu Saahithi</a> 
        · Customer Segmentation using K-Means & Random Forest
        · <a href="https://github.com/bindhusaahithi/Customer-Segmentation-Customer-Lifetime-Value-Prediction" target="_blank">View on GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
