# ==========================================
# Medicine Safety Prediction & Comparison App
# ==========================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from datetime import datetime

# ===============================
# 1. Load dataset
# ===============================
file_path = "medicine_with_expiry_final.csv"
df = pd.read_csv(file_path)

# Clean UPC (prevent scientific notation issues)
if "UPC" in df.columns:
    df["UPC"] = df["UPC"].astype(str).str.replace(r"\.0$", "", regex=True)

# Clean column names
df.columns = (
    df.columns.str.strip()
    .str.replace("Â", "", regex=False)
    .str.replace("°", "C", regex=False)
)
df = df.rename(columns={"Storage Temperature (CC)": "Storage Temperature (C)"})
df["Active Ingredient"] = df["Active Ingredient"].fillna("Unknown")
df["Disease/Use Case"] = df["Disease/Use Case"].fillna("Unknown")

# Target
y = df["Safe/Not Safe"]
le = LabelEncoder()
y = le.fit_transform(y)

# Features
numeric_cols = [
    "Days Until Expiry",
    "Storage Temperature (C)",
    "Dissolution Rate (%)",
    "Disintegration Time (minutes)",
    "Impurity Level (%)",
    "Assay Purity (%)",
    "Warning Labels Present",
]

if df["Warning Labels Present"].dtype == "object":
    df["Warning Labels Present"] = df["Warning Labels Present"].map({"Yes": 1, "No": 0})

X = df[["Active Ingredient", "Disease/Use Case", "UPC"] + numeric_cols]

# Preprocessor
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("text_ing", TfidfVectorizer(max_features=50), "Active Ingredient"),
        ("text_dis", TfidfVectorizer(max_features=50), "Disease/Use Case"),
        ("num", numeric_transformer, numeric_cols),
    ]
)

# Model pipeline
model = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000))]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# ===============================
# 2. Safety rules + hints
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

SAFETY_HINTS = {
    "Days Until Expiry": " (≥30 recommended)",
    "Storage Temperature (C)": " (15–30°C safe range)",
    "Dissolution Rate (%)": " (≥80% recommended)",
    "Disintegration Time (minutes)": " (<30 mins safe)",
    "Impurity Level (%)": " (≤2% safe)",
    "Assay Purity (%)": " (≥90% safe)",
    "Warning Labels Present": " (1 = Yes, 0 = No)",
}

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


# ===============================
# 3. Streamlit UI
# ===============================
st.title("💊 Medicine Safety & Competitor Comparison App")

tab1, tab2, tab3 = st.tabs(["🔍 Medicine Search", "⚖️ Compare Competitor", "📊 Performance Dashboard"])

# ===============================
# TAB 1: Search by UPC or Ingredient
# ===============================
with tab1:
    st.header("🔍 Search Medicine")

    search_option = st.radio("Search by:", ["UPC", "Active Ingredient"])
    if search_option == "UPC":
        upc_input = st.text_input("Enter UPC:")
        if upc_input:
            record = df[df["UPC"].astype(str) == upc_input]
            if not record.empty:
                st.write(record)
            else:
                st.warning("No medicine found with that UPC.")
    else:
        ing_input = st.text_input("Enter Active Ingredient:")
        if ing_input:
            record = df[df["Active Ingredient"].str.contains(ing_input, case=False, na=False)]
            if not record.empty:
                st.write(record)
            else:
                st.warning("No medicine found with that ingredient.")

# ===============================
# TAB 2: Competitor Comparison
# ===============================
with tab2:
    st.header("⚖️ Compare Competitor Medicine")

    comp_upc = st.text_input("Enter Competitor UPC (optional):")
    comp_ing = st.text_input("Enter Competitor Active Ingredient (optional):")
    comp_disease = st.text_input("Enter Disease/Use Case:")

    comp_features = {}
    for col in numeric_cols:
        comp_features[col] = st.number_input(
            f"Enter Competitor Value for {col}{SAFETY_HINTS.get(col,'')}", value=0.0
        )

    if st.button("🔮 Predict & Compare"):
        # Build competitor input
        comp_data = {
            "UPC": comp_upc if comp_upc else "NA",
            "Active Ingredient": comp_ing if comp_ing else "Unknown",
            "Disease/Use Case": comp_disease,
        }
        comp_data.update(comp_features)
        comp_df = pd.DataFrame([comp_data])

        comp_pred = model.predict(comp_df)[0]
        comp_result = le.inverse_transform([comp_pred])[0]

        st.subheader(f"🏷️ Competitor Medicine Prediction: **{comp_result}**")

        # Pick a random standard medicine
        std_sample = df.sample(1, random_state=10)
        std_pred = model.predict(std_sample)[0]
        std_result = le.inverse_transform([std_pred])[0]

        st.write("📌 Comparing with Standard Medicine:")
        st.dataframe(std_sample)

        # Bar chart
        labels = numeric_cols
        comp_values = [comp_data[c] for c in numeric_cols]
        std_values = [std_sample[c].values[0] for c in numeric_cols]

        x = np.arange(len(labels))
        width = 0.35
        comp_colors = ["green" if is_safe(v, c) else "red" for v, c in zip(comp_values, labels)]
        std_colors = ["blue" if is_safe(v, c) else "orange" for v, c in zip(std_values, labels)]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, comp_values, width, label=f"Competitor ({comp_result})", color=comp_colors)
        ax.bar(x + width/2, std_values, width, label=f"Standard ({std_result})", color=std_colors)

        ax.set_ylabel("Values")
        ax.set_title("Medicine Criteria Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.legend()
        st.pyplot(fig)

        # Save log
        log_file = "comparison_results.csv"
        log_entry = {
            "Timestamp": datetime.now(),
            "Competitor UPC": comp_upc,
            "Competitor Ingredient": comp_ing,
            "Competitor Result": comp_result,
            "Standard Ingredient": std_sample["Active Ingredient"].values[0],
            "Standard Result": std_result,
        }
        log_df = pd.DataFrame([log_entry])
        if os.path.exists(log_file):
            log_df.to_csv(log_file, mode="a", header=False, index=False)
        else:
            log_df.to_csv(log_file, index=False)
        st.success("✅ Comparison logged successfully!")

# ===============================
# TAB 3: Performance Dashboard
# ===============================
with tab3:
    st.header("📊 Performance Dashboard")

    log_file = "comparison_results.csv"
    if os.path.exists(log_file):
        history_df = pd.read_csv(log_file)
        history_df["Timestamp"] = pd.to_datetime(history_df["Timestamp"])

        st.write("### 🔍 Recent Comparison Records")
        st.dataframe(history_df.tail(10))

        # Daily test counts
        st.subheader("📈 Daily Test Volume")
        daily_counts = history_df.groupby(history_df["Timestamp"].dt.date).size()
        st.line_chart(daily_counts)

        # Safe vs Not Safe Trend
        st.subheader("⚖️ Safe vs Not Safe Trend")
        safety_trend = history_df.groupby([history_df["Timestamp"].dt.date, "Competitor Result"]).size().unstack(fill_value=0)
        st.bar_chart(safety_trend)

        # Heatmap: Standard vs Competitor
        st.subheader("🏷️ Standard vs Competitor Outcomes")
        outcome_counts = history_df.groupby(["Standard Result", "Competitor Result"]).size().reset_index(name="Count")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(outcome_counts.pivot("Standard Result", "Competitor Result", "Count"),
                    annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("⚠️ No historical data found. Run comparisons first.")
