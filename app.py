import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# Load dataset
# ===============================
file_path = "medicine_dataset.csv"
df = pd.read_csv(file_path, dtype={"UPC": str})
df["UPC"] = df["UPC"].apply(lambda x: str(x).split(".")[0].strip())

# ===============================
# Safety rules
# ===============================
SAFETY_RULES = {
    "Days Until Expiry": {"min": 30},
    "Storage Temperature (C)": {"range": (15, 30)},
    "Dissolution Rate (%)": {"min": 80},
    "Disintegration Time (minutes)": {"max": 30},
    "Impurity Level (%)": {"max": 2},
    "Assay Purity (%)": {"min": 90},
    "Warning Labels Present": {"min": 1}
}

numeric_cols = list(SAFETY_RULES.keys())

# ===============================
# Streamlit UI
# ===============================
st.title("💊 Medicine Safety Comparator")

# --- Search section ---
search_option = st.radio("Search medicine by:", ["UPC", "Active Ingredient"])

medicine = None
if search_option == "UPC":
    upc_input = st.text_input("Enter UPC:")
    if upc_input:
        match = df[df["UPC"] == upc_input]
        if not match.empty:
            medicine = match.iloc[0]
            st.success(f"✅ UPC found → Active Ingredient: {medicine['Active Ingredient']}")
        else:
            st.error("❌ UPC not found in dataset.")
else:
    ing_input = st.text_input("Enter Active Ingredient:")
    if ing_input:
        match = df[df["Active Ingredient"].str.lower() == ing_input.lower()]
        if not match.empty:
            medicine = match.iloc[0]
            st.success(f"✅ Ingredient found → UPC: {medicine['UPC']}")
        else:
            st.error("❌ Ingredient not found in dataset.")

# --- Competitor input ---
st.subheader("🏭 Enter Competitor Medicine Values")

competitor_values = {}
for col in numeric_cols:
    competitor_values[col] = st.number_input(f"{col}:", value=0.0)

# ===============================
# Compare button
# ===============================
if st.button("🔍 Compare"):
    if medicine is None:
        st.error("Please search and select a valid medicine first.")
    else:
        # Values
        user_values = [medicine[col] for col in numeric_cols]
        competitor_values_list = [competitor_values[col] for col in numeric_cols]

        labels = numeric_cols
        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10,6))
        rects1 = ax.bar(x - width/2, user_values, width, label="Standard Medicine")
        rects2 = ax.bar(x + width/2, competitor_values_list, width, label="Competitor Medicine")

        ax.set_ylabel("Values")
        ax.set_title("Medicine Criteria Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.legend()

        # Show graph
        st.pyplot(fig)

        # Safety check report
        st.subheader("📊 Safety Report (Competitor)")
        for col, val in competitor_values.items():
            rule = SAFETY_RULES[col]
            safe = True
            if "min" in rule and val < rule["min"]:
                safe = False
            if "max" in rule and val > rule["max"]:
                safe = False
            if "range" in rule:
                low, high = rule["range"]
                if not (low <= val <= high):
                    safe = False
            status = "✅ Safe" if safe else "❌ Not Safe"
            st.write(f"- {col}: {val} → {status}")
