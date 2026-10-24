import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="AutoValue AI | Vehicle Valuation", page_icon="🚘", layout="wide")

MODEL_PATH = Path(__file__).resolve().parent / "random_forest_car_price_model.pkl"
CURRENCY_SYMBOL = "₹"  # Use "$" if the model target was trained in USD.

def money(value):
    return f"{CURRENCY_SYMBOL}{value:,.2f}"

if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("""
<style>
.stApp {background:#0b1120;color:#e5e7eb}
.block-container {max-width:1350px;padding-top:1.8rem;padding-bottom:3rem}
.hero {background:linear-gradient(120deg,#101d3a,#1d4ed8,#0f766e);padding:30px;border:1px solid #3b82f6;border-radius:22px;margin-bottom:24px}
.brand-row {display:flex;align-items:center;gap:20px;flex-wrap:wrap}
.brand-logo {width:86px;height:86px;display:flex;align-items:center;justify-content:center;background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.3);border-radius:20px;font-size:46px}
.hero h1 {color:#fff;font-size:38px;font-weight:800;margin:5px 0}
.hero p {color:#dbeafe;font-size:15px;line-height:1.6;margin:5px 0}
.eyebrow {color:#bfdbfe;letter-spacing:2px;font-size:11px;font-weight:700}
.result-card {background:linear-gradient(135deg,#132238,#172554);border:1px solid #3b82f6;border-radius:18px;padding:25px;margin:18px 0}
.price-value {color:#34d399;font-size:38px;font-weight:800;margin:10px 0;overflow-wrap:anywhere}
div[data-testid="stMetric"] {background:#111c31;border:1px solid #293b58;border-radius:14px;padding:16px}
div[data-testid="stMetricLabel"] {color:#a5b4fc}
div[data-testid="stMetricValue"] {color:#f8fafc}
section[data-testid="stSidebar"] {background:#0f172a;border-right:1px solid #293b58}
.footer {text-align:center;color:#94a3b8;padding:24px 0;font-size:13px;line-height:1.8}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

if not MODEL_PATH.exists():
    st.error("Model file not found: random_forest_car_price_model.pkl")
    st.info("Place the trained model file in the same folder as app.py.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

with st.sidebar:
    st.markdown("## 🚘 AutoValue AI")
    st.caption("VEHICLE INTELLIGENCE")
    st.divider()
    st.subheader("Dashboard")
    st.write("🤖 Random Forest Model")
    st.write("📊 Vehicle Analytics")
    st.write("💰 Price Valuation")
    st.write("📈 Prediction Trends")
    st.divider()
    st.subheader("Input Features")
    st.caption("Year — manufacturing year")
    st.caption("Condition — vehicle condition score")
    st.caption("Odometer — distance driven in miles")
    st.caption("MMR — market reference price")
    st.divider()
    if st.button("🗑️ Clear Prediction History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

st.markdown("""
<div class="hero">
  <div class="brand-row">
    <div class="brand-logo">🚘</div>
    <div>
      <div class="eyebrow">INTELLIGENT AUTOMOTIVE ANALYTICS</div>
      <h1>AutoValue AI</h1>
      <p>Smart Car Price Prediction &amp; Vehicle Valuation</p>
      <p>Machine-learning estimates, market comparisons, and business insights in one dashboard.</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

history_df = pd.DataFrame(st.session_state.history)
total = len(history_df)
avg = history_df["prediction"].mean() if total else 0
highest = history_df["prediction"].max() if total else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Predictions", f"{total:,}")
k2.metric("Average Estimate", money(avg))
k3.metric("Highest Estimate", money(highest))
k4.metric("ML Algorithm", "Random Forest")

st.divider()
st.header("🚗 Vehicle Valuation")
st.caption("Enter vehicle details to estimate the selling price.")
input_col, info_col = st.columns([1.25, 1])

with input_col:
    st.subheader("Vehicle Details")
    with st.form("vehicle_form"):
        year = st.number_input("Vehicle Manufacturing Year", min_value=1990, max_value=2026, value=2015, step=1)
        condition = st.number_input("Vehicle Condition Score", min_value=0.0, max_value=50.0, value=35.0, step=0.5)
        odometer = st.number_input("Odometer (Miles)", min_value=0, max_value=1_000_000, value=50_000, step=1_000)
        mmr = st.number_input(f"Market Reference Price / MMR ({CURRENCY_SYMBOL})", min_value=0.0, max_value=20_000_000.0, value=150_000.0, step=500.0)
        submitted = st.form_submit_button("🔍 Predict Selling Price", type="primary", use_container_width=True)

with info_col:
    st.subheader("✨ How It Works")
    st.markdown("""
**1. Enter vehicle details**

Add the manufacturing year, condition, odometer, and MMR.

**2. Analyze the vehicle**

The trained Random Forest model processes the four input features.

**3. Review the estimate**

View the estimated selling price and compare it with MMR.

**4. Explore business analytics**

Review trends from predictions generated in this session.
""")
    st.info("The estimate depends on the training data and input quality. It is not a guaranteed selling price.")

if submitted:
    try:
        # Assumes model training order: Year, Condition, Odometer, MMR.
        prediction = float(model.predict(np.array([[year, condition, odometer, mmr]], dtype=float))[0])
        if not np.isfinite(prediction) or prediction < 0:
            st.error("The model returned an invalid price. Check the model and input values.")
        else:
            st.session_state.history.append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "year": int(year),
                "condition": float(condition),
                "odometer": int(odometer),
                "mmr": float(mmr),
                "prediction": prediction,
            })
            st.success("Vehicle valuation completed successfully.")
    except Exception as exc:
        st.error(f"Prediction error: {exc}")
        st.warning("Confirm the model uses the same four features in this exact order: Year, Condition, Odometer, MMR.")

if st.session_state.history:
    latest = st.session_state.history[-1]
    predicted = latest["prediction"]
    latest_mmr = latest["mmr"]
    difference = predicted - latest_mmr
    st.divider()
    st.header("💰 Vehicle Valuation Report")
    st.markdown(f"""
<div class="result-card">
  <p>ESTIMATED SELLING PRICE</p>
  <div class="price-value">{money(predicted)}</div>
  <p>Prediction generated: {latest["time"]}</p>
</div>
""", unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    r1.metric("Market Reference (MMR)", money(latest_mmr))
    r2.metric("Predicted Price", money(predicted))
    r3.metric("Difference vs MMR", money(difference))
    if difference > 0:
        st.info("The estimate is above the entered MMR reference value.")
    elif difference < 0:
        st.info("The estimate is below the entered MMR reference value.")
    else:
        st.info("The estimate equals the entered MMR reference value.")
    st.caption("Difference versus MMR is not actual profit. The predicted price is not guaranteed.")

st.divider()
st.header("📊 Business Analytics")
st.caption("Analytics are based on predictions made in this app session.")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history).copy()
    df["Prediction Number"] = np.arange(1, len(df) + 1)
    df["Difference vs MMR"] = df["prediction"] - df["mmr"]
    df["Difference (%)"] = np.where(df["mmr"] != 0, df["Difference vs MMR"] / df["mmr"] * 100, np.nan)

    st.subheader("Executive Summary")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Vehicles Analyzed", f"{len(df):,}")
    a2.metric("Average MMR", money(df["mmr"].mean()))
    a3.metric("Average Estimate", money(df["prediction"].mean()))
    a4.metric("Average Difference", money(df["Difference vs MMR"].mean()))

    st.subheader("📈 Estimated Price vs Market Reference")
    trend = df.set_index("Prediction Number")[["mmr", "prediction"]].rename(columns={"mmr": "Market Reference", "prediction": "Estimated Selling Price"})
    st.line_chart(trend, x_label="Prediction Number", y_label=f"Price ({CURRENCY_SYMBOL})")
    st.caption("Each point is a separate prediction, not a historical market-price observation.")

    st.subheader("📊 Price Comparison")
    compare = df.set_index("Prediction Number")[["mmr", "prediction"]].rename(columns={"mmr": "MMR", "prediction": "Predicted Price"})
    st.bar_chart(compare, x_label="Prediction Number", y_label=f"Price ({CURRENCY_SYMBOL})")

    st.subheader("🔎 Valuation Position")
    above = int((df["Difference vs MMR"] > 0).sum())
    below = int((df["Difference vs MMR"] < 0).sum())
    equal = int((df["Difference vs MMR"] == 0).sum())
    p1, p2, p3 = st.columns(3)
    p1.metric("Above MMR", above)
    p2.metric("Below MMR", below)
    p3.metric("Equal to MMR", equal)
    positions = pd.DataFrame({"Position": ["Above MMR", "Below MMR", "Equal to MMR"], "Vehicles": [above, below, equal]})
    st.bar_chart(positions, x="Position", y="Vehicles")

    st.subheader("🚘 Vehicle Characteristics")
    chart1, chart2 = st.columns(2)
    with chart1:
        st.markdown("**Odometer vs Predicted Price**")
        mileage = df[["odometer", "prediction"]].rename(columns={"odometer": "Odometer (Miles)", "prediction": "Predicted Price"})
        st.scatter_chart(mileage, x="Odometer (Miles)", y="Predicted Price")
    with chart2:
        st.markdown("**Condition vs Predicted Price**")
        cond = df[["condition", "prediction"]].rename(columns={"condition": "Condition Score", "prediction": "Predicted Price"})
        st.scatter_chart(cond, x="Condition Score", y="Predicted Price")
    st.caption("Charts describe recorded predictions; they do not prove causation or model accuracy.")

    st.divider()
    st.subheader("🧾 Prediction History")
    report = df.rename(columns={"time": "Date & Time", "year": "Vehicle Year", "condition": "Condition Score", "odometer": "Odometer (Miles)", "mmr": "MMR", "prediction": "Predicted Price"})
    cols = ["Date & Time", "Vehicle Year", "Condition Score", "Odometer (Miles)", "MMR", "Predicted Price", "Difference vs MMR", "Difference (%)"]
    st.dataframe(report[cols], use_container_width=True, hide_index=True)
    st.download_button("📥 Download Client Analytics CSV", data=report[cols].to_csv(index=False).encode("utf-8-sig"), file_name="AutoValue_Client_Analytics.csv", mime="text/csv", use_container_width=True)
else:
    st.info("No analytics yet. Enter vehicle details and click Predict Selling Price to start.")

st.divider()
st.markdown("""
<div class="footer">
<strong>🚘 AutoValue AI</strong><br>
Smart Vehicle Valuation &amp; Business Analytics<br>
Powered by Random Forest Regression<br><br>
Predictions are estimates and are not guaranteed vehicle sale prices.
</div>
""", unsafe_allow_html=True)
