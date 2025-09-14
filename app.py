# ==========================================
# Medicine Safety Comparator (Streamlit App)
# ==========================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ===============================
# 1. Load dataset
# ===============================
file_path = "medicine_dataset.csv"   # ✅ Updated to your file
df = pd.read_csv(file_path)

# Clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.replace("Â", "", regex=False)
    .str.replace("°", "C", regex=False)
)
df = df.rename(columns={"Storage Temperature (CC)": "Storage Temperature (C)"})
df["Active Ingredient"] = df["Active Ingredient"].fillna("Unknown")
df["Disease/Use Case"] = df["Disease/Use Case"].fillna("Unknown")

# Ensure UPC is string (avoid scientific notation like 3.18E+11)
if "UPC" in df.columns:
    df["UPC"] = df["UPC"].astype(str)

# Target column
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
    "Warning Labels Present"
]

if df["Warning Labels Present"].dtype == "object":
    df["Warning Labels Present"] = df["Warning Labels Present"].map({"Yes": 1, "No": 0})

X = df[["Active Ingredient", "Disease/Use Case"] + numeric_cols]

# ===============================
# 2. Preprocessor + Model
# ===============================

# Numeric transformer
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Helper function for text column extraction
def get_column(name):
    return FunctionTransformer(lambda x: x[name], validate=False)

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("ing", Pipeline([
            ("selector", get_column("Active Ingredient")),
            ("tfidf", TfidfVectorizer(max_features=50))
        ]), "Active Ingredient"),
        ("dis", Pipeline([
            ("selector", get_column("Disease/Use Case")),
            ("tfidf", TfidfVectorizer(max_features=50))
        ]), "Disease/Use Case"),
        ("num", numeric_transformer, numeric_cols),
    ]
)

# Model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train-test split
if len(np.unique(y)) < 2:
    st.error("❌ Dataset contains only one class in 'Safe/Not Safe'. Logistic Regression needs at least 2.")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)

# ===============================
# 3. Streamlit UI
# ===============================
st.title("💊 Medicine Safety Comparator")

# Search by UPC or Active Ingredient
search_mode = st.radio("Search medicine by:", ["UPC", "Active Ingredient"])

selected_medicine = None
if search_mode == "UPC":
    upc_input = st.text_input("Enter UPC:")
    if upc_input:
        selected_medicine = df[df["UPC"] == upc_input]
elif search_mode == "Active Ingredient":
    ingredient_input = st.selectbox("Select Active Ingredient:", df["Active Ingredient"].unique())
    selected_medicine = df[df["Active Ingredient"] == ingredient_input]

# If medicine is found, auto-fetch details
if selected_medicine is not None and not selected_medicine.empty:
    st.success(f"✅ Found medicine: {selected_medicine.iloc[0]['Active Ingredient']}")
    st.write("Standard Medicine Data:", selected_medicine)

    # Manual competitor input
    st.subheader("🏭 Enter Competitor Medicine Data")
    competitor_data = {}
    for col in numeric_cols:
        competitor_data[col] = st.number_input(
            f"{col}", 
            value=float(selected_medicine.iloc[0][col]),
            step=1.0
        )

    competitor_input = {
        "Active Ingredient": st.text_input("Competitor Active Ingredient"),
        "Disease/Use Case": st.text_input("Competitor Disease/Use Case")
    }
    competitor_input.update(competitor_data)

    comp_df = pd.DataFrame([competitor_input])

    if st.button("🔍 Compare Safety"):
        # Predictions
        comp_pred = model.predict(comp_df)[0]
        comp_result = le.inverse_transform([comp_pred])[0]

        std_pred = model.predict(selected_medicine)[0]
        std_result = le.inverse_transform([std_pred])[0]

        st.write("📌 Standard Medicine Safety:", std_result)
        st.write("📌 Competitor Medicine Safety:", comp_result)

        # ===============================
        # Graph Comparison
        # ===============================
        labels = numeric_cols
        user_values = [competitor_input[col] for col in numeric_cols]
        std_values = [selected_medicine.iloc[0][col] for col in numeric_cols]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        rects1 = ax.bar(x - width/2, std_values, width, label="Standard Medicine", color="green")
        rects2 = ax.bar(x + width/2, user_values, width, label="Competitor", color="red")

        ax.set_ylabel("Values")
        ax.set_title("Medicine Criteria Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.legend()

        # Add labels
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f"{height:.1f}",
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center", va="bottom")
        autolabel(rects1)
        autolabel(rects2)

        st.pyplot(fig)
