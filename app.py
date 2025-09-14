# ==========================================
# Medicine Safety Comparator App (Streamlit)
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

# ===============================
# 2. Ensure at least 2 classes
# ===============================
if df["Safe/Not Safe"].nunique() < 2:
    dummy_row = df.iloc[0].copy()
    dummy_row["Active Ingredient"] = "DummyDrug"
    dummy_row["Safe/Not Safe"] = "Not Safe"   # Force unsafe
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
# 3. Streamlit UI
# ===============================
st.title("💊 Medicine Safety Comparator")
st.write("Compare your medicine against standard dataset medicines.")

# User search options
search_type = st.radio("🔍 Search medicine by:", ["UPC", "Active Ingredient"])
if search_type == "UPC":
    upc_input = st.text_input("Enter UPC code:")
    if upc_input:
        match = df[df["UPC"].astype(str) == str(upc_input)]
        if not match.empty:
            user_tablet = match["Active Ingredient"].values[0]
            st.success(f"UPC found → Active Ingredient: {user_tablet}")
        else:
            st.error("UPC not found in dataset.")
            user_tablet = None
else:
    user_tablet = st.text_input("Enter Active Ingredient:")

user_disease = st.text_input("Enter Disease/Use Case")

# Collect feature inputs
st.subheader("Enter Medicine Test Values")
user_features = {}
for col in numeric_cols:
    user_features[col] = st.number_input(f"{col}", value=0.0)

# Prediction button
if st.button("🔮 Predict Safety"):
    if user_tablet:
        input_data = {"Active Ingredient": user_tablet, "Disease/Use Case": user_disease}
        for col in numeric_cols:
            input_data[col] = user_features[col]
        input_df = pd.DataFrame([input_data])

        pred = model.predict(input_df)[0]
        result = le.inverse_transform([pred])[0]

        st.subheader(f"✅ Prediction Result: **{result}**")

        # Pick a random standard medicine to compare
        other_sample = df.sample(1, random_state=10)
        other_pred = model.predict(other_sample)[0]
        other_result = le.inverse_transform([other_pred])[0]

        st.write(f"📌 Comparing with dataset medicine: **{other_sample['Active Ingredient'].values[0]}** → {other_result}")

        # Bar chart comparison
        labels = numeric_cols
        user_values = [input_data[col] for col in numeric_cols]
        other_values = [other_sample[col].values[0] for col in numeric_cols]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12,6))
        rects1 = ax.bar(x - width/2, user_values, width, label="Your Medicine", color="green" if result=="Safe" else "red")
        rects2 = ax.bar(x + width/2, other_values, width, label="Dataset Medicine", color="blue")

        ax.set_ylabel("Values")
        ax.set_title("Medicine Criteria Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.legend()

        # Label bars
        for rect in rects1 + rects2:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}",
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0,3), textcoords="offset points",
                        ha="center", va="bottom")

        st.pyplot(fig)
    else:
        st.error("Please enter a valid UPC or Active Ingredient.")
