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
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from user_database import init_user_db
from login.py import login_page
from password_reset import password_reset


# Import custom styles
from styles import apply_theme, apply_layout_styles, apply_global_css, set_background, show_logo

# ===============================
# Apply Styles
# ===============================
apply_theme()
apply_layout_styles()
apply_global_css()   # ✅ apply CSS globally

# ===============================
# Page Config
# ===============================
st.set_page_config(page_title="Medicine Safety Comparator", page_icon="💊", layout="wide")
# Initialize login database
init_user_db()

# Show Login Page
username, role = login_page()

# If login fails, stop the app
if username is None:
    st.stop()

# Background + Logo
set_background("bg1.jpg")
show_logo("logo.png")

st.title("💊 Medicine Safety Comparator")

# ===============================
# Sidebar Navigation
# ===============================
with st.sidebar:
    if role == "admin":
    menu = st.sidebar.radio("📌 Navigation", ["📊 Dashboard", "📦 Inventory", "🔑 Change Password"])

elif role == "pharmacist":
    menu = st.sidebar.radio("📌 Navigation", ["🧪 Testing", "📦 Inventory", "🔑 Change Password"])

    st.markdown("<h2 style='color:#2E86C1;'>MedSafe AI</h2>", unsafe_allow_html=True)
    menu = st.radio("📌 Navigation", ["🧪 Testing", "📊 Dashboard", "📦 Inventory"])
    st.markdown("---")
    st.write("ℹ Version 1.0.0")
    st.write("© 2025 MedSafe AI")

# ===============================
# File Paths
# ===============================
MEDICINE_FILE = "medicine_dataset.csv"
INVENTORY_FILE = "inventory.csv"
CONSUMABLES_FILE = "consumables_dataset.csv"   # ✅ missing before
LOG_FILE = "usage_log.csv"

# ===============================
# Load Medicine Dataset
# ===============================
df = pd.read_csv(MEDICINE_FILE, dtype={"UPC": str})
df["UPC"] = df["UPC"].apply(lambda x: str(x).split(".")[0].strip())

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

# Ensure dataset has both classes
if len(np.unique(y)) < 2:
    dummy_row = df.iloc[0].copy()
    dummy_row["Active Ingredient"] = "DummyUnsafe"
    dummy_row["Safe/Not Safe"] = "Not Safe"
    df = pd.concat([df, pd.DataFrame([dummy_row])], ignore_index=True)
    y = df["Safe/Not Safe"]
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

# ===============================
# Train Model
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
    suggestions = []
    for col, val in values.items():
        rule = SAFETY_RULES.get(col, {})
        if "min" in rule and val < rule["min"]:
            suggestions.append(f"Increase *{col}* (min {rule['min']}).")
        if "max" in rule and val > rule["max"]:
            suggestions.append(f"Reduce *{col}* (max {rule['max']}).")
        if "range" in rule:
            low, high = rule["range"]
            if not (low <= val <= high):
                suggestions.append(f"Keep *{col}* within {low}-{high}.")
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
            st.error("⚠ Please enter a valid UPC or Ingredient first.")
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
            st.markdown(f"🏭 Competitor:** {comp_name} | *GST:* {comp_gst} | *Phone:* {comp_phone}")
            st.markdown(f"📍 Address:** {comp_address}")

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
                st.error("⚠ Competitor medicine is NOT SAFE.")
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
                elements.append(Paragraph("<b>⚠ Suggested Improvements:</b>", styles["Heading2"]))
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
                label="⬇ Download PDF Report",
                data=buffer,
                file_name=f"Medicine_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )


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
                col3.metric("⚠ Unsafe", unsafe_count)
                col4.metric("🔥 Top Ingredient", most_common_ing)

                # --- Trend Over Time ---
                st.markdown("<div class='section-header'>📈 Usage Trend Over Time</div>", unsafe_allow_html=True)
                daily_trend = logs.groupby(logs["timestamp"].dt.date).size().reset_index(name="count")
                fig_trend = px.line(
                    daily_trend, x="timestamp", y="count",
                    markers=True,
                    title="Tests Conducted Per Day"
                )
                fig_trend.update_traces(line=dict(width=3, color="#2E86C1"))
                fig_trend.update_layout(title_x=0.5)
                st.plotly_chart(fig_trend, use_container_width=True)

                # --- Recent Logs ---
                st.markdown("<div class='section-header'>📋 Recent Activity</div>", unsafe_allow_html=True)
                st.dataframe(
                    logs.tail(10)[["timestamp", "UPC", "Ingredient", "Competitor", "Result"]],
                    use_container_width=True
                )

                # --- Clear Logs Button ---
                st.markdown("<div class='section-header'>🗑 Manage Logs</div>", unsafe_allow_html=True)
                if st.button("🗑 Clear Logs"):
                    os.remove(LOG_FILE)
                    st.success("✅ Logs cleared successfully. Restart the app to see empty dashboard.")

            else:
                st.info("No data in logs yet. Run some comparisons first.")

        except Exception as e:
            st.error(f"⚠ Could not read logs: {e}")
            st.info("Try clearing or deleting usage_log.csv if the issue persists.")

    else:
        st.info("No logs yet. Run some comparisons to see dashboard data.")




# --- 📦 Inventory Page ---
elif menu == "📦 Inventory":
    st.markdown("<div class='main-title'>📦 Unified Inventory Management</div>", unsafe_allow_html=True)

    # Ensure both files exist
    if not os.path.exists(INVENTORY_FILE):
        pd.DataFrame(columns=["UPC", "Ingredient", "Manufacturer", "Batch", "Stock", "Expiry"]).to_csv(INVENTORY_FILE, index=False)
    if not os.path.exists(CONSUMABLES_FILE):
        pd.DataFrame(columns=[
            "Item Name", "Category", "Material Type", "Sterility Level",
            "Expiry Period (Months)", "Storage Temperature (C)", "Quantity in Stock",
            "Usage Type", "Certification Standard", "UPC", "Safe/Not Safe"
        ]).to_csv(CONSUMABLES_FILE, index=False)

    try:
        # Load datasets
        medicines = pd.read_csv(INVENTORY_FILE)
        consumables = pd.read_csv(CONSUMABLES_FILE)

        # ✅ Normalize medicine column names
        rename_map = {
            "Active Ingredient": "Ingredient",
            "Batch Number": "Batch",
            "Quantity": "Stock",
            "Days Until Expiry": "Days Until Expiry"
        }
        medicines = medicines.rename(columns={k: v for k, v in rename_map.items() if k in medicines.columns})

        # ✅ Add Expiry if missing (using Days Until Expiry)
        if "Expiry" not in medicines.columns and "Days Until Expiry" in medicines.columns:
            today = pd.Timestamp.today()
            medicines["Expiry"] = today + pd.to_timedelta(medicines["Days Until Expiry"], unit="D")

        tab1, tab2 = st.tabs(["💊 Medicines", "🛠 Consumables"])

        # -------------------------
        # 💊 Medicines Tab
        # -------------------------
        with tab1:
            st.markdown("<div class='section-header'>💊 Medicines Inventory</div>", unsafe_allow_html=True)

            # --- KPI Cards ---
            if not medicines.empty:
                total_meds = medicines["Ingredient"].nunique()
                total_stock = medicines["Stock"].sum()
                expiring_soon = medicines[
                    pd.to_datetime(medicines["Expiry"], errors="coerce") <= pd.Timestamp.today() + pd.Timedelta(days=30)
                ]
                expiring_count = len(expiring_soon)

                col1, col2, col3 = st.columns(3)
                col1.metric("💊 Unique Medicines", total_meds)
                col2.metric("📦 Total Stock", total_stock)
                col3.metric("⏳ Expiring Soon", expiring_count)

            # --- Add Medicine ---
            st.markdown("<div class='section-header'>➕ Add / Update Medicine</div>", unsafe_allow_html=True)
            with st.form("add_medicine_form", clear_on_submit=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    upc = st.text_input("UPC")
                with col2:
                    med_name = st.text_input("Ingredient")
                with col3:
                    manufacturer = st.text_input("Manufacturer")

                batch = st.text_input("Batch Number")
                stock = st.number_input("Stock Quantity", min_value=1, step=1)
                expiry = st.date_input("Expiry Date")

                submitted = st.form_submit_button("💾 Save Medicine")
                if submitted:
                    if med_name.strip():
                        # Check if same UPC + Batch exists → update stock
                        mask = (medicines["UPC"] == upc) & (medicines["Batch"] == batch)
                        if medicines[mask].empty:
                            new_entry = pd.DataFrame([[upc, med_name, manufacturer, batch, stock, expiry]],
                                                     columns=["UPC", "Ingredient", "Manufacturer", "Batch", "Stock", "Expiry"])
                            medicines = pd.concat([medicines, new_entry], ignore_index=True)
                        else:
                            medicines.loc[mask, "Stock"] += stock
                            medicines.loc[mask, "Expiry"] = expiry
                        medicines.to_csv(INVENTORY_FILE, index=False)
                        st.success(f"✅ {med_name} saved successfully!")
                    else:
                        st.warning("⚠ Please enter a valid medicine name.")

            # --- View Medicines ---
            st.markdown("<div class='section-header'>📋 Current Medicines</div>", unsafe_allow_html=True)
            if not medicines.empty:
                st.dataframe(medicines, use_container_width=True)
            else:
                st.info("No medicines in inventory yet.")
 

        # -------------------------
        # 🛠 Consumables Tab
        # -------------------------
        with tab2:
            st.markdown("<div class='section-header'>🛠 Consumables Inventory</div>", unsafe_allow_html=True)

            # --- KPI Cards ---
            if not consumables.empty:
                total_items = consumables["Item Name"].nunique()
                total_stock = consumables["Quantity in Stock"].sum()
                expiring_items = consumables[
                    pd.to_numeric(consumables["Expiry Period (Months)"], errors="coerce").fillna(0) <= 1
                ]
                expiring_count = len(expiring_items)

                col1, col2, col3 = st.columns(3)
                col1.metric("🛠 Unique Items", total_items)
                col2.metric("📦 Total Stock", total_stock)
                col3.metric("⏳ Expiring Soon", expiring_count)

            # --- Add Consumable ---
            st.markdown("<div class='section-header'>➕ Add / Update Consumable</div>", unsafe_allow_html=True)
            with st.form("add_consumable_form", clear_on_submit=True):
                col1, col2 = st.columns(2)
                with col1:
                    item_name = st.text_input("Item Name")
                    category = st.text_input("Category")
                    material = st.text_input("Material Type")
                    sterility = st.text_input("Sterility Level")
                with col2:
                    expiry_period = st.number_input("Expiry Period (Months)", min_value=0, step=1)
                    storage_temp = st.number_input("Storage Temp (°C)", step=1)
                    quantity = st.number_input("Quantity in Stock", min_value=1, step=1)
                    upc = st.text_input("UPC")

                usage_type = st.text_input("Usage Type")
                cert = st.text_input("Certification Standard")
                safe_status = st.selectbox("Safe/Not Safe", ["Safe", "Not Safe"])

                submitted = st.form_submit_button("💾 Save Consumable")
                if submitted:
                    if item_name.strip():
                        mask = (consumables["UPC"] == upc)
                        if consumables[mask].empty:
                            new_entry = pd.DataFrame([[item_name, category, material, sterility,
                                                       expiry_period, storage_temp, quantity,
                                                       usage_type, cert, upc, safe_status]],
                                                     columns=consumables.columns)
                            consumables = pd.concat([consumables, new_entry], ignore_index=True)
                        else:
                            consumables.loc[mask, "Quantity in Stock"] += quantity
                            consumables.loc[mask, "Expiry Period (Months)"] = expiry_period
                        consumables.to_csv(CONSUMABLES_FILE, index=False)
                        st.success(f"✅ {item_name} saved successfully!")
                    else:
                        st.warning("⚠ Please enter a valid consumable name.")

            # --- View Consumables ---
            st.markdown("<div class='section-header'>📋 Current Consumables</div>", unsafe_allow_html=True)
            if not consumables.empty:
                st.dataframe(consumables, use_container_width=True)
            else:
                st.info("No consumables in inventory yet.")

    except Exception as e:
        st.error(f"⚠ Could not process inventory: {e}")
        st.info("Try deleting or fixing the CSV files if the issue persists.")

# ===============================
# STEP 6 — PASSWORD RESET PAGE
# ===============================

if menu == "🔑 Change Password":
    st.header("🔑 Change Your Password")

    new_pass = st.text_input("Enter New Password", type="password")
    confirm_pass = st.text_input("Confirm New Password", type="password")

    if st.button("Update Password"):
        if not new_pass or not confirm_pass:
            st.warning("Please fill all fields.")
        elif new_pass != confirm_pass:
            st.error("Passwords do not match!")
        else:
            update_password(username, new_pass)
            st.success("✅ Password updated successfully! Please login again.")
            st.info("Restart the app or refresh the page to continue.")

