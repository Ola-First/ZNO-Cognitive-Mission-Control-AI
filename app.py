import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="ZNO Cognitive Mission Control AI", layout="wide")

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
logo = Image.open("assets/logo.png")

col1, col2 = st.columns([1, 5])
with col1:
    st.image(logo, width=140)
with col2:
    st.title("ZNO Cognitive Mission Control AI")
    st.subheader("AI Decision Assistant for Reducing Managerial Information Overload")

st.divider()

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
data = pd.read_csv("data.csv")

# ---------------------------------------------------
# ADVANCED SIMULATION CONTROLS
# ---------------------------------------------------
st.sidebar.header("Simulation Controls")

workload_multiplier = st.sidebar.slider("Workload Multiplier", 0.8, 1.5, 1.0, 0.05)
email_spike = st.sidebar.slider("Urgent Email Surge", 0, 5, 0)
deadline_spike = st.sidebar.slider("Deadline Pressure Increase", 0, 3, 0)
negativity_spike = st.sidebar.slider("Communication Negativity Spike", 0, 4, 0)

st.sidebar.caption("Simulate cross-channel signal spikes to observe overload evolution.")

# ---------------------------------------------------
# SIGNAL SIMULATION
# ---------------------------------------------------
np.random.seed(42)

data["adjusted_tasks"] = (data["tasks"] * workload_multiplier).round()

data["urgent_emails"] = np.random.randint(0, 6, len(data)) + email_spike
data["approaching_deadlines"] = np.random.randint(0, 4, len(data)) + deadline_spike
data["negative_messages"] = np.random.randint(0, 5, len(data)) + negativity_spike

data["signal_pressure"] = (
    0.4 * data["urgent_emails"] +
    0.4 * data["approaching_deadlines"] +
    0.2 * data["negative_messages"]
)

# ---------------------------------------------------
# RISK ENGINE
# ---------------------------------------------------
data["workload_score"] = data["adjusted_tasks"] / max(data["adjusted_tasks"].max(), 1)
data["hours_score"] = data["hours_logged"] / max(data["hours_logged"].max(), 1)
data["deadline_score"] = data["deadlines_missed"] / max(data["deadlines_missed"].max(), 1)
data["sentiment_risk"] = (1 - data["sentiment_score"]) / 2
data["signal_norm"] = data["signal_pressure"] / max(data["signal_pressure"].max(), 1)

data["risk_score"] = (
    0.30 * data["workload_score"] +
    0.20 * data["hours_score"] +
    0.20 * data["deadline_score"] +
    0.15 * data["sentiment_risk"] +
    0.15 * data["signal_norm"]
)

data["risk_score"] = (data["risk_score"] * 100).round(1)

def categorize_risk(score):
    if score > 70:
        return "High"
    elif score > 40:
        return "Medium"
    return "Low"

data["risk_level"] = data["risk_score"].apply(categorize_risk)

# ---------------------------------------------------
# FORECAST
# ---------------------------------------------------
data["forecast_risk"] = (data["risk_score"] * 1.05).clip(upper=100).round(1)

# ---------------------------------------------------
# ANOMALY DETECTION
# ---------------------------------------------------
data["task_z"] = (
    (data["adjusted_tasks"] - data["adjusted_tasks"].mean())
    / max(data["adjusted_tasks"].std(), 1)
)

data["hours_z"] = (
    (data["hours_logged"] - data["hours_logged"].mean())
    / max(data["hours_logged"].std(), 1)
)

data["anomaly_flag"] = (
    (data["task_z"].abs() > 1.0) |
    (data["hours_z"].abs() > 1.0)
)

anomalies = data[data["anomaly_flag"]]

# ---------------------------------------------------
# ITEM ATTENTION ENGINE
# ---------------------------------------------------
items = pd.DataFrame({
    "item": [
        "Client Escalation Email",
        "Product Launch Task",
        "Budget Review Meeting",
        "Critical Bug Report"
    ],
    "owner": np.random.choice(data["name"], 4),
    "deadline_days": np.random.randint(1, 7, 4),
    "importance": np.random.randint(3, 6, 4),
    "sentiment": np.random.uniform(-0.8, 0.3, 4)
})

items["deadline_pressure"] = 1 / items["deadline_days"]
items["importance_score"] = items["importance"] / items["importance"].max()
items["sentiment_risk"] = (-items["sentiment"]).clip(lower=0)

owner_risk_map = dict(zip(data["name"], data["risk_score"]))
items["owner_risk"] = items["owner"].map(owner_risk_map) / 100

items["attention_score"] = (
    0.35 * items["deadline_pressure"] +
    0.30 * items["importance_score"] +
    0.20 * items["sentiment_risk"] +
    0.15 * items["owner_risk"]
)

items["attention_score"] = (items["attention_score"] * 100).round(1)
items = items.sort_values("attention_score", ascending=False)

# ===================================================
# DISPLAY SECTIONS
# ===================================================

# 1️⃣ Unified Signal Dashboard
st.write("## 📡 Unified Signal Dashboard")
c1, c2, c3 = st.columns(3)
c1.metric("Urgent Emails", int(data["urgent_emails"].sum()))
c2.metric("Deadlines <48h", int(data["approaching_deadlines"].sum()))
c3.metric("Negative Messages", int(data["negative_messages"].sum()))

st.divider()

# 2️⃣ Top Priority Attention Items
st.write("## 🔥 Top Priority Attention Items")
for _, row in items.iterrows():
    if row["attention_score"] > 70:
        st.error(f"{row['item']} — Score: {row['attention_score']}")
    elif row["attention_score"] > 40:
        st.warning(f"{row['item']} — Score: {row['attention_score']}")
    else:
        st.success(f"{row['item']} — Score: {row['attention_score']}")

st.divider()

# 3️⃣ Current Cognitive Risk
st.write("## 🔴 Current Cognitive Risk")
for _, row in data.iterrows():
    if row["risk_level"] == "High":
        st.error(f"{row['name']} — Risk: {row['risk_score']}")
    elif row["risk_level"] == "Medium":
        st.warning(f"{row['name']} — Risk: {row['risk_score']}")
    else:
        st.success(f"{row['name']} — Risk: {row['risk_score']}")

st.divider()

# 4️⃣ Mission Control Status
team_health = 100 - data["risk_score"].mean()
st.write("## 🎯 Mission Control Status")

if team_health > 75:
    st.success(f"Team Health: {round(team_health,1)} / 100 — Stable")
elif team_health > 50:
    st.warning(f"Team Health: {round(team_health,1)} / 100 — Monitor Closely")
else:
    st.error(f"Team Health: {round(team_health,1)} / 100 — Intervention Required")

st.divider()

# 5️⃣ 3-Day Risk Forecast
st.write("## 🔮 3-Day Risk Forecast")
for _, row in data.iterrows():
    if row["forecast_risk"] > 70:
        st.error(f"{row['name']} — Projected Risk: {row['forecast_risk']}")
    elif row["forecast_risk"] > 40:
        st.warning(f"{row['name']} — Projected Risk: {row['forecast_risk']}")
    else:
        st.success(f"{row['name']} — Projected Risk: {row['forecast_risk']}")

st.divider()

# 6️⃣ Anomaly Detection
st.write("## 🧠 Anomaly Detection")
if not anomalies.empty:
    for _, row in anomalies.iterrows():
        st.warning(f"Unusual workload deviation detected for {row['name']}")
else:
    st.success("No statistical anomalies detected.")

st.divider()

# 7️⃣ Sentiment Distribution
st.write("## 💬 Sentiment Distribution")
st.bar_chart(data.set_index("name")["sentiment_score"])

st.divider()

# 8️⃣ Team Data Overview
st.write("## 📊 Team Data Overview")
st.dataframe(data)

st.divider()

# 9️⃣ Executive Summary
st.write("## 🧠 Executive Summary")

if st.button("Generate Executive Summary"):

    high_risk = data[data["risk_level"] == "High"]
    medium_risk = data[data["risk_level"] == "Medium"]
    critical_items = items[items["attention_score"] > 70]

    st.write("### Strategic Insights")

    if not high_risk.empty:
        st.error(f"High overload risk detected for: {', '.join(high_risk['name'])}")

    if not medium_risk.empty:
        st.warning(f"Moderate overload risk observed for: {', '.join(medium_risk['name'])}")

    if not critical_items.empty:
        st.error(f"Immediate attention required for items: {', '.join(critical_items['item'])}")

    if high_risk.empty and medium_risk.empty and critical_items.empty:
        st.success("System operating within acceptable cognitive thresholds.")

    st.write("📌 Recommended Action: Prioritise high-attention items and redistribute workload where appropriate.")
