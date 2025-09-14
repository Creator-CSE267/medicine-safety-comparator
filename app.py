# ==========================================
# app.py – Medicine Safety Comparator (Final)
# ==========================================
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

# ===============================
# 1. Load dataset
# ===============================
file_path = "medicine_dataset.csv"
df = pd.read_csv(file_path, dtype={"UPC": str})
df["UPC"] = df["UPC"].apply(lambda x: str(x).split(".")[0].strip())

# Fill missing values
df["Active Ingredient"] = df["Active Ingredient"].fillna("Unknown")
if "Disease/Use Case" not in df.columns:
    df["Disease/Use Case"] = "Unknown"
else:
    df["Disease/Use Case"] = df["Disease/Use Case"].fillna("Unknown")

# Ensure dataset has at least 2 classes
if df["Safe/Not Safe"].nunique() < 2:
    dummy_row = df.iloc[0].copy()
    dummy_row["Active Ingredient"] = "DummyIngredient"
    dummy_row["UPC"] = "000000000000"
    dummy_row["Safe/Not Safe"] = (
        "Not Safe" if df["Safe/Not Safe"].iloc[0] == "Safe" else "Safe"
    )
    df = pd.concat([df, pd.DataFrame([dummy_row])], ignore_index=True)

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

X = df[["Active Ingredient", "Disease/Use Case"] + numeric_cols]

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
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)

# ===============================
# 2. Streamlit UI
# ===============================
st.set_page_config(page_title="Medicine Safety Comparator", layout="wide")
st.title("💊 Medicine Safety Comparator")
st.write("Compare your competitor’s medicine against your standard dataset.")

# Search Section
st.header("🔎 Search Medicine by UPC or Active Ingredient")

search_mode = st.radio("Search by:", ["UPC", "Active Ingredient"], horizontal=True)

user_upc = None
user_ing = None
selected_row = None

if search_mode == "UPC":
    user_upc = st.text_input("Enter UPC:")
    if user_upc:
        match = df[df["UPC"] == user_upc]
        if not match.empty:
            selected_row = match.iloc[0]
            st.success(f"✅ UPC found → Active Ingredient: {selected_row['Active Ingredient']}")
        else:
            st.error("❌ UPC not found in dataset.")

elif search_mode == "Active Ingredient":
    user_ing = st.text_input("Enter Active Ingredient:")
    if user_ing:
        match = df[df["Active Ingredient"].str.lower() == user_ing.lower()]
        if not match.empty:
            selected_row = match.iloc[0]
            st.success(f"✅ Ingredient found → UPC: {selected_row['UPC']}")
        else:
            st.error("❌ Active Ingredient not found in dataset.")

# ===============================
# 3. Competitor Input Section
# ===============================
st.header("🏭 Enter Competitor Medicine Details")

competitor_features = {}
cols = st.columns(2)
for i, col in enumerate(numeric_cols):
    with cols[i % 2]:
        val = st.number_input(f"{col}", min_value=0.0, step=1.0)
        competitor_features[col] = val

# ===============================
# 4. Compare Button
# ===============================
if selected_row is not None and st.button("🔍 Compare Now"):
    competitor_df = pd.DataFrame(
        [
            {
                "Active Ingredient": selected_row["Active Ingredient"],
                "Disease/Use Case": selected_row["Disease/Use Case"],
                **competitor_features,
            }
        ]
    )
    pred = model.predict(competitor_df)[0]
    result = le.inverse_transform([pred])[0]

    st.subheader("📌 Competitor Prediction Result")
    st.write(f"**{selected_row['Active Ingredient']}** → Prediction: **{result}**")

    # ===============================
    # 5. Comparison Graph
    # ===============================
    st.subheader("📊 Comparison with Standard Medicine")

    labels = numeric_cols
    competitor_values = [competitor_features[col] for col in numeric_cols]
    standard_values = [selected_row[col] for col in numeric_cols]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    rects1 = ax.bar(x - width / 2, competitor_values, width, label="Competitor", color="red")
    rects2 = ax.bar(x + width / 2, standard_values, width, label="Standard", color="green")

    ax.set_ylabel("Values")
    ax.set_title("Medicine Criteria Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.legend()

    st.pyplot(fig)

    # ===============================
    # 6. Performance Dashboard
    # ===============================
    st.subheader("📈 Performance Dashboard")

    safe_count = sum(np.array(competitor_values) <= np.array(standard_values))
    unsafe_count = len(labels) - safe_count

    c1, c2 = st.columns(2)

    with c1:
        st.metric("✅ Safe Criteria Met", safe_count)
        st.metric("❌ Unsafe Criteria", unsafe_count)

    with c2:
        fig2, ax2 = plt.subplots(figsize=(3, 3))
        ax2.pie(
            [safe_count, unsafe_count],
            labels=["Safe", "Unsafe"],
            autopct="%1.1f%%",
            colors=["green", "red"],
        )
        st.pyplot(fig2)
