# ==========================================
# app.py - Medicine Safety Comparator (Streamlit)
# ==========================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ===============================
# 1. Load dataset
# ===============================
DATA_FILE = "medicine_dataset.csv"
LOG_FILE = "competitor_comparison_log.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    return df

df = load_data()

# ===============================
# 2. Safety Rules
# ===============================
SAFETY_RULES = {
    "Days Until Expiry": {"min": 30},
    "Storage Temperature (C)": {"range": (15, 30)},
    "Dissolution Rate (%)": {"min": 80},
    "Disintegration Time (minutes)": {"max": 30},
    "Impurity Level (%)": {"max": 2},
    "Assay Purity (%)": {"min": 90},
    "Warning Labels Present": {"min": 1},
}

# ===============================
# 3. Helper Functions
# ===============================
def is_safe(value, col):
    rule = SAFETY_RULES.get(col, {})
    if "min" in rule and value < rule["min"]:
        return False
    if "max" in rule and value > rule["max"]:
        return False
    if "range" in rule:
        low, high = rule["range"]
        if not (low <= value <= high):
            return False
    return True

def generate_report(values, labels, medicine_name):
    report = []
    for v, col in zip(values, labels):
        safe = is_safe(v, col)
        status = "✅ Safe" if safe else "❌ Not Safe"
        report.append(f"{col}: {v} --> {status}")
    return report

def log_result(upc, ingredient, competitor_values, result):
    log_entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "UPC": upc,
        "Active Ingredient": ingredient,
        "Result": result,
    }
    log_entry.update(competitor_values)

    if os.path.exists(LOG_FILE):
        df_log = pd.read_csv(LOG_FILE)
        df_log = pd.concat([df_log, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        df_log = pd.DataFrame([log_entry])

    df_log.to_csv(LOG_FILE, index=False)

# ===============================
# 4. Streamlit App UI
# ===============================
st.set_page_config(page_title="💊 Medicine Safety Comparator", layout="wide")
st.title("💊 Medicine Safety Comparator")
st.write("Compare competitor medicines vs standard dataset medicines.")

# Search Mode
mode = st.radio("Search by:", ["UPC", "Active Ingredient"])

if mode == "UPC":
    upc_input = st.text_input("Enter UPC:")
    selected_row = df[df["UPC"].astype(str) == upc_input]
elif mode == "Active Ingredient":
    ingredient_input = st.text_input("Enter Active Ingredient:")
    selected_row = df[df["Active Ingredient"].str.lower() == ingredient_input.lower()]

if selected_row is not None and not selected_row.empty:
    st.success("✅ Standard medicine found!")
    st.write(selected_row)

    # Extract standard medicine values
    labels = list(SAFETY_RULES.keys())
    standard_values = [selected_row.iloc[0][col] for col in labels]

    # Competitor Input
    st.subheader("🏭 Enter Competitor Medicine Values")
    competitor_values = {}
    for col in labels:
        competitor_values[col] = st.number_input(
            f"{col}:", value=float(standard_values[labels.index(col)]), step=1.0
        )

    # Compare Button
    if st.button("🔍 Compare"):
        competitor_vals_list = [competitor_values[col] for col in labels]

        # Safety reports
        st.subheader("📊 Safety Reports")
        st.write("**Standard Medicine**")
        for line in generate_report(standard_values, labels, "Standard Medicine"):
            st.write(line)

        st.write("**Competitor Medicine**")
        comp_report = generate_report(competitor_vals_list, labels, "Competitor Medicine")
        for line in comp_report:
            st.write(line)

        # Final Result
        final_result = (
            "Safe" if all(is_safe(v, c) for v, c in zip(competitor_vals_list, labels)) else "Not Safe"
        )
        st.subheader(f"✅ Final Prediction: {final_result}")

        # Log Results
        log_result(
            upc=selected_row.iloc[0]["UPC"],
            ingredient=selected_row.iloc[0]["Active Ingredient"],
            competitor_values=competitor_values,
            result=final_result,
        )

        # Plot Comparison
        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width/2, standard_values, width, label="Standard", color="green")
        ax.bar(x + width/2, competitor_vals_list, width, label="Competitor", color="red")
        ax.set_ylabel("Values")
        ax.set_title("Medicine Criteria Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.legend()
        st.pyplot(fig)

# ===============================
# 5. Performance Dashboard
# ===============================
st.header("📈 Performance Dashboard")
if os.path.exists(LOG_FILE):
    log_df = pd.read_csv(LOG_FILE)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tests", len(log_df))
    col2.metric("Safe Medicines", sum(log_df["Result"] == "Safe"))
    col3.metric("Not Safe Medicines", sum(log_df["Result"] == "Not Safe"))

    # Daily test counts
    log_df["Date"] = pd.to_datetime(log_df["Timestamp"]).dt.date
    daily_counts = log_df.groupby("Date").size()

    fig, ax = plt.subplots(figsize=(8, 4))
    daily_counts.plot(kind="bar", ax=ax)
    ax.set_title("Daily Tests Performed")
    st.pyplot(fig)

    # Safe vs Not Safe Pie Chart
    fig, ax = plt.subplots()
    log_df["Result"].value_counts().plot.pie(
        autopct="%1.1f%%", startangle=90, ax=ax, colors=["green", "red"]
    )
    ax.set_ylabel("")
    ax.set_title("Safe vs Not Safe Distribution")
    st.pyplot(fig)

else:
    st.info("No logs yet. Run a comparison first!")
