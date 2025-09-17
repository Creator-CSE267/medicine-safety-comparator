# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from datetime import datetime
from PIL import Image

# ===============================
# Page Design
# ===============================
st.set_page_config(page_title="Medicine Safety Comparator", page_icon="💊", layout="wide")

# Background
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://cdn.vectorstock.com/i/1000v/42/27/modern-medicine-science-background-vector-21144227.jpg");
        background-size: cover;
        background-attachment: fixed;
    }
    .block-container {
        background-color: white;  /* solid instead of transparent */
        padding: 20px 40px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ===============================
# Sidebar Navigation
# ===============================
with st.sidebar:
    # Logo
    if os.path.exists("logo.png"):
        logo = Image.open("logo.png")
        st.image(logo, width=120)
    st.markdown("<h2 style='color:#2E86C1;'>MedSafe AI</h2>", unsafe_allow_html=True)

    # Menu
    menu = st.radio("📌 Navigation", ["🧪 Testing", "📊 Dashboard", "📦 Inventory"])

    st.markdown("---")
    st.write("ℹ️ Version 1.0.0")
    st.write("© 2025 MedSafe AI")

# ===============================
# Load dataset
# ===============================
DATA_FILE = "medicine_dataset.csv"
LOG_FILE = "usage_log.csv"

df = pd.read_csv(DATA_FILE, dtype={"UPC": str})
df["UPC"] = df["UPC"].apply(lambda x: str(x).split(".")[0].strip())

# Fill missing values
df["Active Ingredient"] = df["Active Ingredient"].fillna("Unknown")
if "Disease/Use Case" not in df.columns:
    df["Disease/Use Case"] = "Unknown"
else:
    df["Disease/Use Case"] = df["Disease/Use Case"].fillna("Unknown")

# Target
if "Safe/Not Safe" not in df.columns:
    df["Safe/Not Safe"] = "Safe"

y = df["Safe/Not Safe"]
le = LabelEncoder()
y = le.fit_transform(y)

# Add dummy row if dataset has only 1 class
if len(np.unique(y)) < 2:
    dummy_row = df.iloc[0].copy()
    dummy_row["Active Ingredient"] = "DummyUnsafe"
    dummy_row["Safe/Not Safe"] = "Not Safe"
    df = pd.concat([df, pd.DataFrame([dummy_row])], ignore_index=True)
    y = df["Safe/Not Safe"]
    y = le.fit_transform(y)

# Feature columns
numeric_cols = [
    "Days Until Expiry",
    "Storage Temperature (C)",
    "Dissolution Rate (%)",
    "Disintegration Time (minutes)",
    "Impurity Level (%)",
    "Assay Purity (%)",
    "Warning Labels Present"
]

if df["Warning Labels Present"].dtype == "object":
    df["Warning Labels Present"] = df["Warning Labels Present"].map({"Yes": 1, "No": 0})

X = df[["Active Ingredient", "Disease/Use Case"] + numeric_cols]

# ===============================
# Train model
# ===============================
def train_model(X, y):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("text_ing", TfidfVectorizer(max_features=50), "Active Ingredient"),
            ("text_dis", TfidfVectorizer(max_features=50), "Disease/Use Case"),
            ("num", numeric_transformer, numeric_cols)
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    return model

model = train_model(X, y)

# ===============================
# Pages
# ===============================

# --- 🧪 Testing Page ---
if menu == "🧪 Testing":
    st.header("🧪 Medicine Safety Testing")
    st.subheader("🔍 Search by UPC or Active Ingredient")

    col1, col2 = st.columns(2)
    with col1:
        upc_input = st.text_input("Enter UPC:")
    with col2:
        ingredient_input = st.text_input("Enter Active Ingredient:")

    selected_row = None
    if upc_input:
        match = df[df["UPC"] == upc_input]
        if not match.empty:
            selected_row = match.iloc[0]
            ingredient_input = selected_row["Active Ingredient"]
            st.success(f"✅ UPC found → Active Ingredient: {ingredient_input}")
        else:
            st.error("❌ UPC not found in dataset.")
    elif ingredient_input:
        match = df[df["Active Ingredient"].str.lower() == ingredient_input.lower()]
        if not match.empty:
            selected_row = match.iloc[0]
            upc_input = selected_row["UPC"]
            st.success(f"✅ Ingredient found → UPC: {upc_input}")
        else:
            st.error("❌ Ingredient not found in dataset.")

    st.subheader("🏭 Competitor Medicine Entry")
    competitor_values = {}
    for col in numeric_cols:
        competitor_values[col] = st.number_input(f"{col}:", value=0.0)

    if st.button("🔎 Compare"):
        if selected_row is None:
            st.error("⚠️ Please enter a valid UPC or Ingredient first.")
        else:
            input_data = {"Active Ingredient": ingredient_input, "Disease/Use Case": "Unknown"}
            for col in numeric_cols:
                input_data[col] = competitor_values[col]
            competitor_df = pd.DataFrame([input_data])

            pred = model.predict(competitor_df)[0]
            result = le.inverse_transform([pred])[0]

            base_values = [selected_row[col] for col in numeric_cols]
            comp_values = [competitor_values[col] for col in numeric_cols]

            st.success(f"✅ Competitor Prediction: {result}")

            # Comparison chart
            x = np.arange(len(numeric_cols))
            width = 0.35
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(x - width/2, base_values, width, label="Standard Medicine", color="green")
            ax.bar(x + width/2, comp_values, width, label="Competitor Medicine", color="red")
            ax.set_xticks(x)
            ax.set_xticklabels(numeric_cols, rotation=30, ha="right")
            ax.set_title("Medicine Criteria Comparison")
            ax.legend()
            st.pyplot(fig)

            # Log
            log_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "UPC": upc_input,
                "Ingredient": ingredient_input,
                "Result": result
            }
            log_df = pd.DataFrame([log_entry])
            if not os.path.exists(LOG_FILE):
                log_df.to_csv(LOG_FILE, index=False)
            else:
                log_df.to_csv(LOG_FILE, mode="a", header=False, index=False)

# --- 📊 Dashboard Page ---
elif menu == "📊 Dashboard":
    st.header("📊 Performance Dashboard")

    if os.path.exists(LOG_FILE):
        logs = pd.read_csv(LOG_FILE)
        logs["timestamp"] = pd.to_datetime(logs["timestamp"])

        st.subheader("Recent Usage Logs")
        st.dataframe(logs.tail(10))

        st.subheader("Safety Prediction Summary")
        st.bar_chart(logs["Result"].value_counts())

        st.subheader("Daily Usage Trend")
        daily_trend = logs.groupby(logs["timestamp"].dt.date).size()
        st.line_chart(daily_trend)

        st.subheader("Most Frequently Compared Medicines")
        st.bar_chart(logs["Ingredient"].value_counts().head(5))

        st.subheader("Competitor Safety Success Rate (%)")
        success_rate = (logs["Result"].value_counts(normalize=True) * 100).round(2)
        st.dataframe(success_rate)
    else:
        st.info("No logs yet. Run some comparisons to see dashboard data.")

# --- 📦 Inventory Page ---
elif menu == "📦 Inventory":
    st.header("📦 Medicine Inventory")

    st.write("Browse the dataset currently loaded into the app.")
    st.dataframe(df)

    st.write("### Dataset Overview")
    st.write(f"Total Medicines: {len(df)}")
    st.write("Active Ingredients Distribution:")
    st.bar_chart(df["Active Ingredient"].value_counts().head(10))
