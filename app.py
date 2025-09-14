import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ===============================
# 1. Load dataset
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("medicine_dataset.csv", dtype={"UPC": str})
    df["UPC"] = df["UPC"].apply(lambda x: str(x).split(".")[0].strip())
    df["Active Ingredient"] = df["Active Ingredient"].fillna("Unknown")
    if "Disease/Use Case" not in df.columns:
        df["Disease/Use Case"] = "Unknown"
    else:
        df["Disease/Use Case"] = df["Disease/Use Case"].fillna("Unknown")
    return df

df = load_data()

# ===============================
# 2. Train model (cached)
# ===============================
@st.cache_resource
def train_model(df):
    y = df["Safe/Not Safe"]
    le = LabelEncoder()
    y = le.fit_transform(y)

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

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    # Model metrics
    y_pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0)
    }

    return model, metrics

model, metrics = train_model(df)

# ===============================
# 3. Sidebar Performance Metrics
# ===============================
st.sidebar.title("📊 Model Performance")
for k, v in metrics.items():
    st.sidebar.write(f"**{k}:** {v:.2f}")

# ===============================
# 4. Streamlit App UI
# ===============================
st.title("💊 Medicine Safety Comparator")

# Input options
st.subheader("🔍 Search by UPC or Active Ingredient")
col1, col2 = st.columns(2)

with col1:
    upc_input = st.text_input("Enter UPC")

with col2:
    ingredient_input = st.text_input("Enter Active Ingredient")

selected_row = None
if upc_input:
    if upc_input in df["UPC"].values:
        selected_row = df[df["UPC"] == upc_input].iloc[0]
        st.success(f"✅ UPC found → Active Ingredient: {selected_row['Active Ingredient']}")
    else:
        st.error("UPC not found in dataset")

elif ingredient_input:
    if ingredient_input in df["Active Ingredient"].values:
        selected_row = df[df["Active Ingredient"] == ingredient_input].iloc[0]
        st.success(f"✅ Ingredient found → UPC: {selected_row['UPC']}")
    else:
        st.error("Ingredient not found in dataset")

# ===============================
# 5. Competitor Input
# ===============================
st.subheader("🏭 Competitor Medicine Entry")

competitor_data = {}
numeric_cols = [
    "Days Until Expiry",
    "Storage Temperature (C)",
    "Dissolution Rate (%)",
    "Disintegration Time (minutes)",
    "Impurity Level (%)",
    "Assay Purity (%)",
    "Warning Labels Present"
]

for col in numeric_cols:
    competitor_data[col] = st.number_input(f"Enter {col}", value=0.0)

# ===============================
# 6. Compare Button
# ===============================
if st.button("🔎 Compare with Standard"):
    if selected_row is not None:
        # Build competitor dataframe
        competitor_df = pd.DataFrame([{
            "Active Ingredient": selected_row["Active Ingredient"],
            "Disease/Use Case": selected_row["Disease/Use Case"],
            **competitor_data
        }])

        # Prediction
        pred = model.predict(competitor_df)[0]
        result = "✅ SAFE" if pred == 1 else "❌ NOT SAFE"

        st.subheader("📊 Comparison Result")
        st.write(f"**Prediction:** {result}")

        # Side-by-side comparison
        standard_values = selected_row[numeric_cols].to_dict()

        comparison_df = pd.DataFrame({
            "Criteria": numeric_cols,
            "Standard Medicine": [standard_values[c] for c in numeric_cols],
            "Competitor Medicine": [competitor_data[c] for c in numeric_cols]
        })

        st.dataframe(comparison_df)

        # Graph
        st.subheader("📈 Performance Dashboard")
        fig, ax = plt.subplots(figsize=(10, 5))
        comparison_df.plot(x="Criteria", kind="bar", ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning("⚠️ Please enter a valid UPC or Active Ingredient before comparing.")
