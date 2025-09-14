# ================================================
# Medicine Safety Comparator App (with Dashboard)
# ================================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
file_path = "medicine_dataset.csv"
df = pd.read_csv(file_path, dtype={"UPC": str})
df["UPC"] = df["UPC"].apply(lambda x: str(x).split(".")[0].strip())

# Clean dataset
df["Active Ingredient"] = df["Active Ingredient"].fillna("Unknown")
if "Disease/Use Case" not in df.columns:
    df["Disease/Use Case"] = "Unknown"
else:
    df["Disease/Use Case"] = df["Disease/Use Case"].fillna("Unknown")

# Encode target
y = df["Safe/Not Safe"]
le = LabelEncoder()
y = le.fit_transform(y)

# Numeric features
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

# Preprocessor
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

# Model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure 2 classes exist
if len(set(y_train)) > 1:
    model.fit(X_train, y_train)

# ===============================
# 2. Streamlit UI
# ===============================
st.title("💊 Medicine Safety Comparator")
st.write("Compare competitor medicines with standard medicines and view performance trends.")

# UPC or Ingredient Input
search_mode = st.radio("Search by:", ["UPC", "Active Ingredient"])

selected_upc, selected_ingredient = None, None

if search_mode == "UPC":
    upc_input = st.text_input("Enter UPC:")
    if upc_input:
        match = df[df["UPC"] == upc_input]
        if not match.empty:
            selected_upc = upc_input
            selected_ingredient = match.iloc[0]["Active Ingredient"]
            st.success(f"✅ UPC found → Active Ingredient: {selected_ingredient}")
        else:
            st.error("❌ UPC not found in dataset.")

else:
    ingredient_input = st.text_input("Enter Active Ingredient:")
    if ingredient_input:
        match = df[df["Active Ingredient"].str.lower() == ingredient_input.lower()]
        if not match.empty:
            selected_ingredient = ingredient_input
            selected_upc = match.iloc[0]["UPC"]
            st.success(f"✅ Ingredient found → UPC: {selected_upc}")
        else:
            st.error("❌ Ingredient not found in dataset.")

# ===============================
# 3. Competitor Inputs
# ===============================
st.subheader("🧪 Enter Competitor Medicine Values")

SAFETY_HINTS = {
    "Days Until Expiry": " (≥30 recommended)",
    "Storage Temperature (C)": " (15–30°C safe range)",
    "Dissolution Rate (%)": " (≥80% recommended)",
    "Disintegration Time (minutes)": " (<30 mins safe)",
    "Impurity Level (%)": " (≤2% safe)",
    "Assay Purity (%)": " (≥90% safe)",
    "Warning Labels Present": " (1 = Yes, 0 = No)"
}

competitor_data = {}
for col in numeric_cols:
    competitor_data[col] = st.number_input(f"{col}{SAFETY_HINTS[col]}", value=0.0)

# ===============================
# 4. Compare Button
# ===============================
if st.button("🔍 Compare Medicines"):
    if selected_ingredient:
        # Build competitor input
        competitor_row = {"Active Ingredient": selected_ingredient, "Disease/Use Case": "Unknown"}
        competitor_row.update(competitor_data)
        competitor_df = pd.DataFrame([competitor_row])

        # Standard medicine (from dataset)
        standard = df[df["Active Ingredient"] == selected_ingredient].iloc[0]

        # Predictions
        pred = model.predict(competitor_df)[0]
        result = le.inverse_transform([pred])[0]

        st.subheader("📊 Comparison Result")
        st.write(f"**Competitor Medicine Prediction:** {result}")

        # ===============================
        # Radar Chart Comparison
        # ===============================
        labels = numeric_cols
        competitor_values = [competitor_row[col] for col in numeric_cols]
        standard_values = [standard[col] for col in numeric_cols]

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        competitor_values += competitor_values[:1]
        standard_values += standard_values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
        ax.plot(angles, competitor_values, "o-", label="Competitor", color="red")
        ax.fill(angles, competitor_values, alpha=0.25, color="red")
        ax.plot(angles, standard_values, "o-", label="Standard", color="green")
        ax.fill(angles, standard_values, alpha=0.25, color="green")
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.legend()
        st.pyplot(fig)

        # ===============================
        # Safety Report
        # ===============================
        st.subheader("📝 Safety Report")
        for col, val in competitor_data.items():
            st.write(f"- {col}: {val} (Standard: {standard[col]})")

        # ===============================
        # Save to history (CSV)
        # ===============================
        log_entry = {
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "UPC": selected_upc,
            "Active Ingredient": selected_ingredient,
            "Competitor Result": result
        }
        history_file = "comparison_history.csv"
        try:
            history_df = pd.read_csv(history_file)
            history_df = pd.concat([history_df, pd.DataFrame([log_entry])], ignore_index=True)
        except FileNotFoundError:
            history_df = pd.DataFrame([log_entry])
        history_df.to_csv(history_file, index=False)

# ===============================
# 5. Performance Dashboard
# ===============================
st.subheader("📈 Performance Dashboard")
try:
    history_df = pd.read_csv("comparison_history.csv")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Comparisons", len(history_df))
        daily_counts = history_df["Date"].str[:10].value_counts().sort_index()
        fig, ax = plt.subplots()
        daily_counts.plot(kind="bar", ax=ax)
        ax.set_title("Daily Comparisons")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with col2:
        pie_data = history_df["Competitor Result"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%", startangle=90)
        ax.set_title("Safe vs Not Safe")
        st.pyplot(fig)

except FileNotFoundError:
    st.info("No comparisons made yet. Run a test first.")
