# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from datetime import datetime
from PIL import Image
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import io
import streamlit as st
from style import apply_theme, apply_layout_styles

apply_theme()
apply_layout_styles()

# Initialize session state for theme persistence
if "theme_choice" not in st.session_state:
    st.session_state.theme_choice = "Light"   # Default theme
if "custom_theme" not in st.session_state:
    st.session_state.custom_theme = {
        "text_color": "#000000",
        "metric_bg": "#f0f0f0",
        "button_text": "#000000",
        "button_bg": "#e0e0e0",
        "header_color": "#000000",
    }

# Preset themes
THEMES = {
    "Light": {
        "text_color": "#000000",
        "metric_bg": "#f9f9f9",
        "button_text": "#000000",
        "button_bg": "#e0e0e0",
        "header_color": "#000000",
    },
    "Dark": {
        "text_color": "#FFFFFF",
        "metric_bg": "#333333",
        "button_text": "#FFFFFF",
        "button_bg": "#444444",
        "header_color": "#FFFFFF",
    }
}

# Sidebar theme selector
st.sidebar.header("🎨 Theme Settings")
theme_choice = st.sidebar.radio(
    "Choose Theme",
    ["Light", "Dark", "Custom"],
    index=["Light", "Dark", "Custom"].index(st.session_state.theme_choice)
)

# Save user choice
st.session_state.theme_choice = theme_choice

# Apply chosen theme
if theme_choice in THEMES:
    THEME = THEMES[theme_choice]
else:
    # Custom theme via color pickers
    custom = st.session_state.custom_theme
    custom["text_color"]   = st.sidebar.color_picker("Text Color", custom["text_color"])
    custom["metric_bg"]    = st.sidebar.color_picker("KPI Card Background", custom["metric_bg"])
    custom["button_text"]  = st.sidebar.color_picker("Button Text Color", custom["button_text"])
    custom["button_bg"]    = st.sidebar.color_picker("Button Background", custom["button_bg"])
    custom["header_color"] = st.sidebar.color_picker("Header Color", custom["header_color"])
    st.session_state.custom_theme = custom
    THEME = custom

# Apply global CSS
st.markdown(f"""
    <style>
        html, body, [class*="css"] {{
            color: {THEME['text_color']} !important;
        }}

        .stMetric {{
            background: {THEME['metric_bg']};
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.08);
            color: {THEME['text_color']} !important;
        }}

        .stButton>button {{
            width: 100%;
            margin-bottom: 10px;
            border-radius: 10px;
            height: 3em;
            font-weight: bold;
            color: {THEME['button_text']} !important;
            background-color: {THEME['button_bg']} !important;
        }}

        .stDataFrame, .stTable {{
            color: {THEME['text_color']} !important;
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: {THEME['header_color']} !important;
        }}
    </style>
""", unsafe_allow_html=True)

# ===============================
# Page Design
# ===============================
st.set_page_config(page_title="Medicine Safety Comparator", page_icon="💊", layout="wide")

# --- Background ---
def set_background(image_file):
    import base64
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            background-attachment: fixed;
            color: #FFFFFF;
        }}
        .block-container {{
            background: transparent !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Example background
set_background("bg4.jpg")

# --- Logo Centered ---
if os.path.exists("logo.png"):
    logo = Image.open("logo.png")
    st.image(logo, width=120)
    st.markdown(
        """
        <style>
        .logo-container {
            display: flex;
            justify-content: center;
            margin-bottom: 10px;
        }
        .logo-container img {
            width: 180px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="logo-container"><img src="logo.png"></div>', unsafe_allow_html=True)

st.title("💊 Medicine Safety Comparator")

# ===============================
# Sidebar Navigation
# ===============================
with st.sidebar:
    st.markdown("<h2 style='color:#2E86C1;'>MedSafe AI</h2>", unsafe_allow_html=True)

    menu = st.radio("📌 Navigation", ["🧪 Testing", "📊 Dashboard", "📦 Inventory"])

    st.markdown("---")
    st.write("ℹ️ Version 1.0.0")
    st.write("© 2025 MedSafe AI")

# ===============================
# Load dataset
# ===============================
DATA_FILE = "medicine_dataset.csv"
LOG_FILE = "usage_log.csv"
INVENTORY_FILE = "inventory.csv"

df = pd.read_csv(DATA_FILE, dtype={"UPC": str})
df["UPC"] = df["UPC"].apply(lambda x: str(x).split(".")[0].strip())

# Fill missing values
df["Active Ingredient"] = df["Active Ingredient"].fillna("Unknown")
if "Disease/Use Case" not in df.columns:
    df["Disease/Use Case"] = "Unknown"
else:
    df["Disease/Use Case"] = df["Disease/Use Case"].fillna("Unknown")

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
# Safety Rules
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

def suggest_improvements(values):
    """Check competitor values against safety rules and suggest fixes."""
    suggestions = []
    for col, val in values.items():
        rule = SAFETY_RULES.get(col, {})
        if "min" in rule and val < rule["min"]:
            suggestions.append(f"Increase **{col}** (min {rule['min']}).")
        if "max" in rule and val > rule["max"]:
            suggestions.append(f"Reduce **{col}** (max {rule['max']}).")
        if "range" in rule:
            low, high = rule["range"]
            if not (low <= val <= high):
                suggestions.append(f"Keep **{col}** within {low}-{high}.")
    return suggestions

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
    comp_name = st.text_input("Competitor Name")
    comp_gst = st.text_input("GST Number")
    comp_address = st.text_area("Address")
    comp_phone = st.text_input("Phone Number")

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

            # Show competitor details
            st.markdown(f"**🏭 Competitor:** {comp_name} | **GST:** {comp_gst} | **Phone:** {comp_phone}")
            st.markdown(f"**📍 Address:** {comp_address}")

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

            # Suggestions if unsafe
            suggestions = []
            if result.lower() == "not safe":
                st.error("⚠️ Competitor medicine is NOT SAFE.")
                suggestions = suggest_improvements(competitor_values)
                if suggestions:
                    st.markdown("### 🔧 Suggested Improvements")
                    for s in suggestions:
                        st.write(f"- {s}")

            # Log
            log_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "UPC": upc_input,
                "Ingredient": ingredient_input,
                "Competitor": comp_name,
                "Result": result
            }
            log_df = pd.DataFrame([log_entry])
            if not os.path.exists(LOG_FILE):
                log_df.to_csv(LOG_FILE, index=False)
            else:
                log_df.to_csv(LOG_FILE, mode="a", header=False, index=False)

            # --- PDF Report Download ---
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.pagesizes import A4
            import io

            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            elements = []

            # --- Add Logo ---
            if os.path.exists("logo.png"):
                elements.append(RLImage("logo.png", width=100, height=100))
                elements.append(Spacer(1, 12))

            # --- Title & Date ---
            elements.append(Paragraph("💊 Medicine Safety Comparison Report", styles["Title"]))
            elements.append(Spacer(1, 12))
            elements.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
            elements.append(Spacer(1, 12))

            # --- Standard Medicine ---
            elements.append(Paragraph("<b>Standard Medicine</b>", styles["Heading2"]))
            elements.append(Paragraph(f"UPC: {upc_input}", styles["Normal"]))
            elements.append(Paragraph(f"Ingredient: {ingredient_input}", styles["Normal"]))
            elements.append(Spacer(1, 12))

            # --- Competitor Medicine ---
            elements.append(Paragraph("<b>Competitor Medicine</b>", styles["Heading2"]))
            elements.append(Paragraph(f"Name: {comp_name}", styles["Normal"]))
            elements.append(Paragraph(f"GST Number: {comp_gst}", styles["Normal"]))
            elements.append(Paragraph(f"Address: {comp_address}", styles["Normal"]))
            elements.append(Paragraph(f"Phone: {comp_phone}", styles["Normal"]))
            elements.append(Spacer(1, 12))

            # --- Prediction ---
            elements.append(Paragraph("<b>Prediction Result</b>", styles["Heading2"]))
            if result.lower() == "safe":
                elements.append(Paragraph(f"<font color='green'><b>{result}</b></font>", styles["Normal"]))
            else:
                elements.append(Paragraph(f"<font color='red'><b>{result}</b></font>", styles["Normal"]))
            elements.append(Spacer(1, 12))

            # --- Suggestions if Not Safe ---
            if result.lower() == "not safe" and suggestions:
                elements.append(Paragraph("<b>⚠️ Suggested Improvements:</b>", styles["Heading2"]))
                for s in suggestions:
                    elements.append(Paragraph(f"- {s}", styles["Normal"]))
                elements.append(Spacer(1, 12))

            # --- Add Comparison Chart ---
            chart_buffer = io.BytesIO()
            fig.savefig(chart_buffer, format="png")
            chart_buffer.seek(0)
            elements.append(RLImage(chart_buffer, width=400, height=250))
            elements.append(Spacer(1, 12))

            # --- Build PDF ---
            doc.build(elements)
            buffer.seek(0)

            # --- Streamlit Download Button ---
            st.download_button(
                label="⬇️ Download PDF Report",
                data=buffer,
                file_name=f"Medicine_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )

# --- 📊 Dashboard Page ---
from style import apply_global_css

# --- 📊 Dashboard Page ---
elif menu == "📊 Dashboard":
    apply_global_css()   # ✅ apply styling
    st.markdown("<div class='main-title'>📊 Medicine Safety Analytics Dashboard</div>", unsafe_allow_html=True)

    if os.path.exists(LOG_FILE):
        try:
            logs = pd.read_csv(LOG_FILE, on_bad_lines="skip")
            logs["timestamp"] = pd.to_datetime(logs["timestamp"], errors="coerce")

            if not logs.empty:
                # --- KPI Cards ---
                total_tests = len(logs)
                safe_count = logs["Result"].str.lower().eq("safe").sum()
                unsafe_count = logs["Result"].str.lower().eq("not safe").sum()
                most_common_ing = logs["Ingredient"].mode()[0] if "Ingredient" in logs.columns else "N/A"

                st.markdown("<div class='section-header'>📌 Key Performance Indicators</div>", unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("🧪 Total Tests", total_tests)
                col2.metric("✅ Safe", safe_count)
                col3.metric("⚠️ Unsafe", unsafe_count)
                col4.metric("🔥 Most Common Ingredient", most_common_ing)

                # --- Competitor Safety Gauge ---
                st.markdown("<div class='section-header'>📊 Competitor Safety Rate</div>", unsafe_allow_html=True)
                safety_rate = (safe_count / total_tests * 100) if total_tests > 0 else 0
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=safety_rate,
                    title={"text": "Safety %"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "green"},
                        "steps": [
                            {"range": [0, 40], "color": "red"},
                            {"range": [40, 70], "color": "orange"},
                            {"range": [70, 100], "color": "lightgreen"}
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 4},
                            "thickness": 0.75,
                            "value": safety_rate
                        }
                    }
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)

                # --- Trend Over Time ---
                st.markdown("<div class='section-header'>📈 Daily Usage Trend</div>", unsafe_allow_html=True)
                daily_trend = logs.groupby(logs["timestamp"].dt.date).size().reset_index(name="count")
                fig_trend = px.line(daily_trend, x="timestamp", y="count", markers=True, title="Tests Over Time")
                st.plotly_chart(fig_trend, use_container_width=True)

                # --- Safe vs Unsafe Pie ---
                st.markdown("<div class='section-header'>🟢 Safe vs 🔴 Unsafe Distribution</div>", unsafe_allow_html=True)
                result_counts = logs["Result"].value_counts()
                fig_pie = px.pie(values=result_counts.values, names=result_counts.index, hole=0.4)
                st.plotly_chart(fig_pie, use_container_width=True)

                # --- Top Competitors ---
                if "Competitor" in logs.columns:
                    st.markdown("<div class='section-header'>🏭 Top 5 Compared Competitors</div>", unsafe_allow_html=True)
                    top_comp = logs["Competitor"].value_counts().head(5).reset_index()
                    top_comp.columns = ["Competitor", "Count"]
                    fig_bar = px.bar(top_comp, x="Competitor", y="Count", text="Count")
                    st.plotly_chart(fig_bar, use_container_width=True)

                # --- Recent Logs ---
                st.markdown("<div class='section-header'>📋 Recent Activity</div>", unsafe_allow_html=True)
                st.dataframe(logs.tail(10), use_container_width=True)

                # --- Clear Logs Button ---
                st.markdown("<div class='section-header'>🗑️ Manage Logs</div>", unsafe_allow_html=True)
                if st.button("🗑️ Clear Logs"):
                    os.remove(LOG_FILE)
                    st.success("✅ Logs cleared successfully. Restart the app to see empty dashboard.")

            else:
                st.info("No data in logs yet. Run some comparisons first.")

        except Exception as e:
            st.error(f"⚠️ Could not read logs: {e}")
            st.info("Try clearing or deleting `usage_log.csv` if the issue persists.")

    else:
        st.info("No logs yet. Run some comparisons to see dashboard data.")


# --- 📦 Inventory Page ---
elif menu == "📦 Inventory":
    st.markdown("<div class='main-title'>📦 Medicine Inventory Management</div>", unsafe_allow_html=True)

    # Ensure inventory file exists
    if not os.path.exists(INVENTORY_FILE):
        pd.DataFrame(columns=["Medicine", "Stock", "Expiry"]).to_csv(INVENTORY_FILE, index=False)

    try:
        inventory = pd.read_csv(INVENTORY_FILE)

        # --- Add Medicine ---
        st.markdown("<div class='section-header'>➕ Add Medicine to Inventory</div>", unsafe_allow_html=True)
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            with st.form("add_medicine_form", clear_on_submit=True):
                med_name = st.text_input("Medicine Name")
                stock = st.number_input("Stock Quantity", min_value=1, step=1)
                expiry = st.date_input("Expiry Date")
                submitted = st.form_submit_button("➕ Add Medicine")
                if submitted:
                    new_entry = pd.DataFrame([[med_name, stock, expiry]], columns=["Medicine", "Stock", "Expiry"])
                    inventory = pd.concat([inventory, new_entry], ignore_index=True)
                    inventory.to_csv(INVENTORY_FILE, index=False)
                    st.success(f"✅ {med_name} added successfully!")
            st.markdown("</div>", unsafe_allow_html=True)

        # --- View Inventory ---
        st.markdown("<div class='section-header'>📋 Current Inventory</div>", unsafe_allow_html=True)
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            if not inventory.empty:
                st.dataframe(inventory, use_container_width=True)
            else:
                st.info("No medicines in inventory yet.")
            st.markdown("</div>", unsafe_allow_html=True)

        # --- Remove Medicine ---
        st.markdown("<div class='section-header'>🗑️ Remove Medicine</div>", unsafe_allow_html=True)
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            if not inventory.empty:
                med_to_remove = st.selectbox("Select Medicine to Remove", inventory["Medicine"].unique())
                if st.button("🗑️ Remove Selected"):
                    inventory = inventory[inventory["Medicine"] != med_to_remove]
                    inventory.to_csv(INVENTORY_FILE, index=False)
                    st.success(f"✅ {med_to_remove} removed successfully!")
            else:
                st.info("No medicines available to remove.")
            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Could not read inventory: {e}")
        st.info("Try deleting or fixing `inventory.csv` if the issue persists.")
