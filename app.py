# ==========================================
# Medicine Safety Comparator App (Final)
# ==========================================
import pandas as pd
import streamlit as st
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

# Clean up
df["Active Ingredient"] = df["Active Ingredient"].fillna("Unknown")
if "Disease/Use Case" not in df.columns:
    df["Disease/Use Case"] = "Unknown"
else:
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

# Model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)

# ===============================
# 2. Streamlit App
# ===============================
st.title("💊 Medicine Safety Comparator")
st.markdown("Compare your test medicine against standard dataset medicines.")

# ===============================
# 3. Search Section (UPC ↔ Ingredient)
# ===============================
search_type = st.radio("Search by:", ["UPC", "Active Ingredient"])

user_tablet = None
user_upc = None

if search_type == "UPC":
    upc_input = st.text_input("Enter UPC code:")
    if upc_input:
        match = df[df["UPC"] == upc_input.strip()]
        if not match.empty:
            user_tablet = match["Active Ingredient"].values[0]
            user_upc = upc_input.strip()
            st.success(f"✅ UPC found → Active Ingredient: {user_tablet}")
        else:
            st.error("❌ UPC not found in dataset.")

elif search_type == "Active Ingredient":
    ing_input = st.text_input("Enter Active Ingredient:")
    if ing_input:
        match = df[df["Active Ingredient"].str.lower() == ing_input.lower()]
        if not match.empty:
            user_upc = match["UPC"].values[0]
            user_tablet = match["Active Ingredient"].values[0]
            st.success(f"✅ Ingredient found → UPC: {user_upc}")
        else:
            st.error("❌ Ingredient not found in dataset.")

# ===============================
# 4. Safety Rules
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
# 5. User Input Competitor Values
# ===============================
if user_tablet and user_upc:
    st.subheader(f"📊 Enter Test Values for: {user_tablet} (UPC: {user_upc})")

    user_features = {}
    for col in numeric_cols:
        val = st.number_input(f"{col}:", value=0.0)
        user_features[col] = val

    # Input row
    input_data = {"Active Ingredient": user_tablet, "Disease/Use Case": "Unknown"}
    for col in numeric_cols:
        input_data[col] = user_features[col]
    input_df = pd.DataFrame([input_data])

    # Predict
    pred = model.predict(input_df)[0]
    result = le.inverse_transform([pred])[0]
    st.success(f"✅ Prediction for your test input: **{result}**")

    # ===============================
    # 6. Pick random dataset medicine to compare
    # ===============================
    other_sample = df.sample(1, random_state=10)
    other_pred = model.predict(other_sample)[0]
    other_result = le.inverse_transform([other_pred])[0]
    other_name = other_sample["Active Ingredient"].values[0]

    st.info(f"📌 Comparing against dataset medicine: **{other_name}** → {other_result}")

    # ===============================
    # 7. Comparison Graph
    # ===============================
    labels = numeric_cols
    user_values = [input_data[col] for col in numeric_cols]
    other_values = [other_sample[col].values[0] for col in numeric_cols]

    x = np.arange(len(labels))
    width = 0.35

    user_colors = ["green" if is_safe(v, c) else "red" for v, c in zip(user_values, labels)]
    other_colors = ["blue" if is_safe(v, c) else "orange" for v, c in zip(other_values, labels)]

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, user_values, width, label=f"Your Medicine ({result})", color=user_colors)
    rects2 = ax.bar(x + width/2, other_values, width, label=f"Dataset Medicine ({other_result})", color=other_colors)

    ax.set_ylabel("Values")
    ax.set_title("Medicine Criteria Comparison with Safety Thresholds")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}",
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom")
    autolabel(rects1)
    autolabel(rects2)

    st.pyplot(fig)

    # ===============================
    # 8. Performance Dashboard (Mini)
    # ===============================
    st.subheader("📈 Performance Dashboard")
    safe_counts = df["Safe/Not Safe"].value_counts()
    fig2, ax2 = plt.subplots()
    ax2.pie(safe_counts, labels=safe_counts.index, autopct="%1.1f%%", startangle=90)
    ax2.set_title("Overall Dataset Safety Distribution")
    st.pyplot(fig2)
