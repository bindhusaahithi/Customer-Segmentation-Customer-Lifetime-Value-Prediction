# 🎯 Customer Segmentation & CLV Prediction

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://customer-segmentation-customer-lifetime-value-prediction.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/bindhusaahithi)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/bindhusaahithi)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/bindhu-saahithi-naralashetty-yogendranath/)

---

## 🚀 Live Demo

**👉 [Try the app here](https://customer-segmentation-customer-lifetime-value-prediction.streamlit.app)**

Enter customer RFM values and instantly see which segment they belong to with business recommendations!

---

## 📌 Project Overview

Segmented customers using **RFM Analysis** (Recency, Frequency, Monetary) and predicted **Customer Lifetime Value** to drive smarter marketing and retention strategies.

Built on a real-world UK e-commerce dataset with **500,000+ transactions**.

---

## 🎯 Key Results

- Segmented **4,338 unique customers** into 4 distinct groups
- Achieved **R² score of 0.45** on CLV prediction
- Identified top revenue-generating countries and customer behavior patterns
- Built and deployed a live interactive web app

---

## 🛠️ Tech Stack

`Python` `Pandas` `NumPy` `Scikit-learn` `Matplotlib` `Seaborn` `Streamlit`

**Models Used:**
- K-Means Clustering (customer segmentation)
- Random Forest Regressor (CLV prediction)
- StandardScaler (feature preprocessing)

---

## 📊 Customer Segments

| Segment | Description | Strategy |
|---|---|---|
| 💎 High Value | Frequent buyers, high spend | Loyalty rewards, VIP treatment |
| 😴 At-Risk | Inactive recently | Win-back campaigns, discounts |
| 🌱 New/Potential | Low frequency, promising | Welcome series, first purchase offer |
| 📦 Regular | Moderate activity | Upsell, restock alerts |

---

## 📁 Project Structure

```
Customer-Segmentation/
├── app.py                    ← Streamlit web app
├── kmeans_model.pkl          ← Saved K-Means model
├── scaler.pkl                ← Saved StandardScaler
├── rf_model.pkl              ← Saved Random Forest model
├── requirements.txt
├── data/
│   └── Online Retail.xlsx    ← Dataset
├── notebooks/
│   └── customer_segmentation.ipynb
└── visuals/                  ← Generated charts
```

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/bindhusaahithi/Customer-Segmentation-Customer-Lifetime-Value-Prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📈 Sample Visualizations

- Monthly Revenue Trend
- Top Countries by Revenue
- Elbow Method for Optimal Clusters
- Customer Segments Scatter Plot
- Feature Importance Analysis

---

## 👩‍💻 About

Built by **Bindhu Saahithi** — Data Science Graduate Student at UMass Dartmouth

🌍 Open to Data Scientist & Analyst roles in **USA & UK**

[![GitHub](https://img.shields.io/badge/GitHub-bindhusaahithi-181717?style=flat&logo=github)](https://github.com/bindhusaahithi)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Bindhu_Saahithi-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/bindhu-saahithi-naralashetty-yogendranath/)
[![Kaggle](https://img.shields.io/badge/Kaggle-bindhusaahithi-20BEFF?style=flat&logo=kaggle)](https://www.kaggle.com/bindhusaahithi)
