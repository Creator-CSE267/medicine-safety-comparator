# app.py  (MongoDB-only, Option A)
import os
import io
from datetime import datetime
from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Mongo
import certifi
from pymongo import MongoClient
from bson import ObjectId

# Report
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# App modules (keep these files in your repo)
from login import login_router
from user_database import init_user_db
from password_reset import password_reset

# Styling helpers (keep these files in repo)
from styles import apply_theme, apply_layout_styles, apply_global_css, set_background, show_logo

# ----------------------- CONFIG -----------------------
SESSION_TIMEOUT_SECONDS = 30 * 60

# ----------------------- MONGO CONNECT -----------------------
@st.cache_resource
def get_db():
    """
    Connect to MongoDB Atlas using certifi CA bundle for TLS verification.
    Expects Streamlit Secrets:
      [MONGO]
      URI = "<mongodb+srv://...>"
      DBNAME = "your_db_name"
    """
    try:
        uri = st.secrets["MONGO"]["URI"]
        dbname = st.secrets["MONGO"]["DBNAME"]
    except Exception:
        uri = os.getenv("MONGO_URI")
        dbname = os.getenv("MONGO_DBNAME")

    if not uri or not dbname:
        st.error("MongoDB secrets missing. Add MONGO.URI and MONGO.DBNAME to Streamlit secrets.")
        st.stop()

    client_opts = {
        "serverSelectionTimeoutMS": 20000,
        "connectTimeoutMS": 20000,
        "tls": True,
        "tlsCAFile": certifi.where(),
    }

    client = MongoClient(uri, **client_opts)
    # fail fast
    try:
        client.admin.command("ping")
    except Exception as e:
        st.error(f"Could not connect to MongoDB: {e}")
        raise
    return client[dbname]

# initialize DB & collections
db = get_db()
medicines_col = db["medicines"]       # ML dataset
inventory_col = db["inventory"]
consumables_col = db["consumables"]
logs_col = db["usage_log"]
users_col = db["users"]               # for reference if you need it

# ----------------------- INIT USERS -----------------------
init_user_db()

# ----------------------- SESSION DEFAULTS -----------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None
if "role" not in st.session_state:
    st.session_state["role"] = None
if "last_active" not in st.session_state:
    st.session_state["last_active"] = None

def session_is_timed_out() -> bool:
    last = st.session_state.get("last_active")
    if not last:
        return False
    return (datetime.now() - datetime.fromisoformat(last)).total_seconds() > SESSION_TIMEOUT_SECONDS

if st.session_state["authenticated"] and session_is_timed_out():
    st.warning("Session timed out. Login again.")
    st.session_state["authenticated"] = False
    st.rerun()

# ----------------------- LOGIN -----------------------
if not st.session_state["authenticated"]:
    login_router()
    st.stop()

# auth success
username = st.session_state["username"]
role = st.session_state["role"]
st.session_state["last_active"] = datetime.now().isoformat()

# ----------------------- PAGE LAYOUT / THEME -----------------------
st.set_page_config(page_title="Medicine Safety Comparator", page_icon="💊", layout="wide")
apply_theme()
apply_layout_styles()
apply_global_css()
set_background("bg1.jpg")
show_logo("logo.png")
st.title("💊 Medicine Safety Comparator")

# ----------------------- DB HELPER FUNCTIONS -----------------------

# --- Medicines (ML dataset) ---
def load_medicines_df() -> pd.DataFrame:
    docs = list(medicines_col.find({}))
    if not docs:
        return pd.DataFrame()
    for d in docs:
        d["_id"] = str(d["_id"])
    df = pd.DataFrame(docs)
    # ensure UPC string
    if "UPC" in df.columns:
        df["UPC"] = df["UPC"].astype(str)
    return df

def replace_medicines_from_df(df: pd.DataFrame):
    """Replace entire medicines collection with provided DataFrame (use for initial upload)."""
    medicines_col.delete_many({})
    if df.empty:
        return
    records = df.to_dict(orient="records")
    for r in records:
        # convert NaN -> None and numpy types
        for k,v in r.items():
            if pd.isna(v):
                r[k] = None
            elif isinstance(v, (np.integer, np.floating)):
                r[k] = v.item()
    if records:
        medicines_col.insert_many(records)

# --- Inventory ---
def load_inventory_df() -> pd.DataFrame:
    docs = list(inventory_col.find({}))
    if not docs:
        cols = ["UPC","Ingredient","Manufacturer","Batch","Stock","Expiry","_id"]
        return pd.DataFrame(columns=cols)
    for d in docs:
        d["_id"] = str(d["_id"])
    df = pd.DataFrame(docs)
    if "Expiry" in df.columns:
        df["Expiry"] = pd.to_datetime(df["Expiry"], errors="coerce")
    if "Stock" in df.columns:
        df["Stock"] = pd.to_numeric(df["Stock"], errors="coerce").fillna(0).astype(int)
    return df

def save_inventory_item(doc: dict):
    q = {"UPC": doc.get("UPC"), "Batch": doc.get("Batch")}
    existing = inventory_col.find_one(q)
    # Normalize expiry
    if "Expiry" in doc and doc["Expiry"]:
        try:
            doc["Expiry"] = pd.to_datetime(doc["Expiry"]).isoformat()
        except Exception:
            doc["Expiry"] = str(doc["Expiry"])
    if existing:
        # increment stock if provided
        upd = {}
        if "Stock" in doc:
            try:
                upd["Stock"] = int(existing.get("Stock", 0)) + int(doc.get("Stock", 0))
            except:
                upd["Stock"] = doc.get("Stock")
        if "Expiry" in doc:
            upd["Expiry"] = doc["Expiry"]
        if upd:
            inventory_col.update_one({"_id": existing["_id"]}, {"$set": upd})
        return str(existing["_id"])
    else:
        inventory_col.insert_one(doc)
        return None

def update_inventory_by_id(id_str: str, fields: dict):
    inventory_col.update_one({"_id": ObjectId(id_str)}, {"$set": fields})

def delete_inventory_by_id(id_str: str):
    inventory_col.delete_one({"_id": ObjectId(id_str)})

# --- Consumables ---
def load_consumables_df() -> pd.DataFrame:
    docs = list(consumables_col.find({}))
    if not docs:
        cols = ["Item Name","Category","Material Type","Sterility Level",
                "Expiry Period (Months)","Storage Temperature (C)","Quantity in Stock",
                "Usage Type","Certification Standard","UPC","Safe/Not Safe","_id"]
        return pd.DataFrame(columns=cols)
    for d in docs:
        d["_id"] = str(d["_id"])
    df = pd.DataFrame(docs)
    if "Quantity in Stock" in df.columns:
        df["Quantity in Stock"] = pd.to_numeric(df["Quantity in Stock"], errors="coerce").fillna(0).astype(int)
    return df

def save_consumable_item(doc: dict):
    key = {"UPC": doc.get("UPC")} if doc.get("UPC") else None
    existing = consumables_col.find_one(key) if key else None
    if existing:
        consumables_col.update_one({"_id": existing["_id"]}, {"$set": doc})
        return str(existing["_id"])
    else:
        consumables_col.insert_one(doc)
        return None

def update_consumable_by_id(id_str: str, fields: dict):
    consumables_col.update_one({"_id": ObjectId(id_str)}, {"$set": fields})

def delete_consumable_by_id(id_str: str):
    consumables_col.delete_one({"_id": ObjectId(id_str)})

# --- Logs ---
def append_log(entry: dict):
    if "timestamp" in entry:
        try:
            entry["timestamp"] = pd.to_datetime(entry["timestamp"]).isoformat()
        except:
            entry["timestamp"] = datetime.now().isoformat()
    else:
        entry["timestamp"] = datetime.now().isoformat()
    logs_col.insert_one(entry)

def load_logs_df(limit=5000) -> pd.DataFrame:
    docs = list(logs_col.find({}).sort([("_id", -1)]).limit(limit))
    if not docs:
        return pd.DataFrame(columns=["timestamp","UPC","Ingredient","Competitor","Result","_id"])
    for d in docs:
        d["_id"] = str(d["_id"])
        if "timestamp" in d:
            d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    return pd.DataFrame(docs)

def clear_logs_in_db():
    logs_col.delete_many({})

# ----------------------- ML MODEL SETUP -----------------------
# Load medicines dataset from DB and build model. If empty, show uploader (one-time).
med_df = load_medicines_df()
if med_df.empty:
    st.warning("Medicine dataset for model training is empty.")
    st.info("Upload a CSV to populate the model dataset (one-time). Columns expected: UPC, Active Ingredient, Disease/Use Case, Days Until Expiry, etc.")
    upload = st.file_uploader("Upload medicines CSV", type=["csv"])
    if upload:
        try:
            uploaded_df = pd.read_csv(upload, dtype={"UPC": str})
            if "UPC" in uploaded_df.columns:
                uploaded_df["UPC"] = uploaded_df["UPC"].astype(str).apply(lambda x: str(x).split(".")[0].strip())
            replace_medicines_from_df(uploaded_df)
            st.success("Medicines dataset uploaded. Reload the app to train the model.")
            st.stop()
        except Exception as e:
            st.error(f"Upload failed: {e}")
            st.stop()
    st.stop()

# Prepare df for training
df = med_df.copy()
df["Active Ingredient"] = df.get("Active Ingredient", pd.Series(["Unknown"] * len(df))).fillna("Unknown")
if "Disease/Use Case" not in df.columns:
    df["Disease/Use Case"] = "Unknown"
else:
    df["Disease/Use Case"] = df["Disease/Use Case"].fillna("Unknown")
if "Safe/Not Safe" not in df.columns:
    df["Safe/Not Safe"] = "Safe"

y = df["Safe/Not Safe"]
le = LabelEncoder()
y = le.fit_transform(y)

# ensure at least two classes for training
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

if "Warning Labels Present" in df.columns and df["Warning Labels Present"].dtype == "object":
    df["Warning Labels Present"] = df["Warning Labels Present"].map({"Yes": 1, "No": 0})

X = df[["Active Ingredient", "Disease/Use Case"] + [c for c in numeric_cols if c in df.columns]]

def train_model(X, y):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("text_ing", TfidfVectorizer(max_features=50), "Active Ingredient"),
            ("text_dis", TfidfVectorizer(max_features=50), "Disease/Use Case"),
            ("num", numeric_transformer, [c for c in numeric_cols if c in X.columns])
        ]
    )
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    # if insufficient numeric columns exist, model will still train on text features
    X_train, X_test, y_train, y_test = train_test_split(X.fillna(0), y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model(X, y)

# ----------------------- SAFETY RULES -----------------------
SAFETY_RULES = {
    "Days Until Expiry": {"min": 30},
    "Storage Temperature (C)": {"range": (15, 30)},
    "Dissolution Rate (%)": {"min": 80},
    "Disintegration Time (minutes)": {"max": 30},
    "Impurity Level (%)": {"max": 2},
    "Assay Purity (%)": {"min": 90},
    "Warning Labels Present": {"min": 1}
}

def suggest_improvements(values: dict):
    suggestions = []
    for col, val in values.items():
        rule = SAFETY_RULES.get(col, {})
        try:
            v = float(val)
        except:
            v = val
        if "min" in rule and isinstance(v, (int, float)) and v < rule["min"]:
            suggestions.append(f"Increase *{col}* (min {rule['min']}).")
        if "max" in rule and isinstance(v, (int, float)) and v > rule["max"]:
            suggestions.append(f"Reduce *{col}* (max {rule['max']}).")
        if "range" in rule and isinstance(v, (int, float)):
            low, high = rule["range"]
            if not (low <= v <= high):
                suggestions.append(f"Keep *{col}* within {low}-{high}.")
    return suggestions

# ----------------------- SIDEBAR -----------------------
def render_avatar(name: Optional[str], size=72):
    avatar_path_png = os.path.join("avatars", f"{name}.png")
    avatar_path_jpg = os.path.join("avatars", f"{name}.jpg")
    if os.path.exists(avatar_path_png):
        st.sidebar.image(avatar_path_png, width=size)
        return
    if os.path.exists(avatar_path_jpg):
        st.sidebar.image(avatar_path_jpg, width=size)
        return
    initials = "".join([p[0] for p in (name or "User").split()][:2]).upper()
    circle_html = f"""
    <div style="
        width:{size}px;height:{size}px;border-radius:50%;
        background: linear-gradient(135deg,#2E86C1,#5DADE2);
        display:flex;align-items:center;justify-content:center;
        font-weight:700;color:white;font-size:{size//2}px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    ">{initials}</div>
    """
    st.sidebar.markdown(circle_html, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h3 style='color:#2E86C1;margin-bottom:6px;'>MedSafe AI</h3>", unsafe_allow_html=True)
    render_avatar(username, size=72)
    st.sidebar.write(f"**{username}**")
    st.sidebar.write(f"Role: **{role}**")
    st.sidebar.markdown("---")

    if st.sidebar.button("Logout 🔒"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.session_state["role"] = None
        st.session_state["last_active"] = None
        st.success("Logged out.")
        st.experimental_rerun()

    if role == "admin":
        menu = st.sidebar.radio("📌 Navigation", ["📊 Dashboard", "📦 Inventory", "🔑 Change Password"])
    elif role == "pharmacist":
        menu = st.sidebar.radio("📌 Navigation", ["🧪 Testing", "📦 Inventory", "🔑 Change Password"])
    else:
        menu = st.sidebar.radio("📌 Navigation", ["📦 Inventory"])

    st.sidebar.markdown("---")
    st.sidebar.write("© 2025 MedSafe AI")

# ----------------------- PAGES -----------------------

# --- Testing Page ---
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
        match = df[df["UPC"] == upc_input] if "UPC" in df.columns else pd.DataFrame()
        if not match.empty:
            selected_row = match.iloc[0]
            ingredient_input = selected_row["Active Ingredient"]
            st.success(f"✅ UPC found → Active Ingredient: {ingredient_input}")
        else:
            st.error("❌ UPC not found in dataset.")
    elif ingredient_input:
        match = df[df["Active Ingredient"].str.lower() == ingredient_input.lower()] if "Active Ingredient" in df.columns else pd.DataFrame()
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

            base_values = [selected_row.get(col, 0) for col in numeric_cols]
            comp_values = [competitor_values[col] for col in numeric_cols]

            st.success(f"✅ Competitor Prediction: {result}")

            st.markdown(f"🏭 Competitor:** {comp_name} | *GST:* {comp_gst} | *Phone:* {comp_phone}")
            st.markdown(f"📍 Address:** {comp_address}")

            x = np.arange(len(numeric_cols))
            width = 0.35
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(x - width/2, base_values, width, label="Standard Medicine")
            ax.bar(x + width/2, comp_values, width, label="Competitor Medicine")
            ax.set_xticks(x)
            ax.set_xticklabels(numeric_cols, rotation=30, ha="right")
            ax.set_title("Medicine Criteria Comparison")
            ax.legend()
            st.pyplot(fig)

            suggestions = []
            if result.lower() == "not safe":
                st.error("⚠ Competitor medicine is NOT SAFE.")
                suggestions = suggest_improvements(competitor_values)
                if suggestions:
                    st.markdown("### 🔧 Suggested Improvements")
                    for s in suggestions:
                        st.write(f"- {s}")

            # Log to DB
            append_log({
                "timestamp": datetime.now().isoformat(),
                "UPC": upc_input,
                "Ingredient": ingredient_input,
                "Competitor": comp_name,
                "Result": result
            })

            # PDF report
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            elements = []
            if os.path.exists("logo.png"):
                elements.append(RLImage("logo.png", width=100, height=100))
                elements.append(Spacer(1, 12))
            elements.append(Paragraph("💊 Medicine Safety Comparison Report", styles["Title"]))
            elements.append(Spacer(1, 12))
            elements.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
            elements.append(Spacer(1, 12))
            elements.append(Paragraph("<b>Standard Medicine</b>", styles["Heading2"]))
            elements.append(Paragraph(f"UPC: {upc_input}", styles["Normal"]))
            elements.append(Paragraph(f"Ingredient: {ingredient_input}", styles["Normal"]))
            elements.append(Spacer(1, 12))
            elements.append(Paragraph("<b>Competitor Medicine</b>", styles["Heading2"]))
            elements.append(Paragraph(f"Name: {comp_name}", styles["Normal"]))
            elements.append(Paragraph(f"GST Number: {comp_gst}", styles["Normal"]))
            elements.append(Paragraph(f"Address: {comp_address}", styles["Normal"]))
            elements.append(Paragraph(f"Phone: {comp_phone}", styles["Normal"]))
            elements.append(Spacer(1, 12))
            elements.append(Paragraph("<b>Prediction Result</b>", styles["Heading2"]))
            elements.append(Paragraph(f"<b>{result}</b>", styles["Normal"]))
            elements.append(Spacer(1, 12))
            if suggestions:
                elements.append(Paragraph("<b>Suggested Improvements</b>", styles["Heading2"]))
                for s in suggestions:
                    elements.append(Paragraph(f"- {s}", styles["Normal"]))
                elements.append(Spacer(1, 12))
            chart_buffer = io.BytesIO()
            fig.savefig(chart_buffer, format="png")
            chart_buffer.seek(0)
            elements.append(RLImage(chart_buffer, width=400, height=250))
            doc.build(elements)
            buffer.seek(0)
            st.download_button("⬇ Download PDF Report", data=buffer, file_name=f"Medicine_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")

# --- Dashboard Page ---
elif menu == "📊 Dashboard":
    apply_global_css()
    st.markdown("<div class='main-title'>📊 Medicine Safety Analytics Dashboard</div>", unsafe_allow_html=True)
    logs = load_logs_df(limit=5000)
    if not logs.empty:
        total_tests = len(logs)
        safe_count = logs["Result"].str.lower().eq("safe").sum() if "Result" in logs.columns else 0
        unsafe_count = logs["Result"].str.lower().eq("not safe").sum() if "Result" in logs.columns else 0
        most_common_ing = logs["Ingredient"].mode()[0] if "Ingredient" in logs.columns and not logs["Ingredient"].mode().empty else "N/A"

        st.markdown("<div class='section-header'>📌 Key Performance Indicators</div>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🧪 Total Tests", total_tests)
        col2.metric("✅ Safe", safe_count)
        col3.metric("⚠ Unsafe", unsafe_count)
        col4.metric("🔥 Top Ingredient", most_common_ing)

        st.markdown("<div class='section-header'>📈 Usage Trend Over Time</div>", unsafe_allow_html=True)
        daily_trend = logs.groupby(logs["timestamp"].dt.date).size().reset_index(name="count")
        fig_trend = px.line(daily_trend, x="timestamp", y="count", markers=True, title="Tests Conducted Per Day")
        st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown("<div class='section-header'>📋 Recent Activity</div>", unsafe_allow_html=True)
        st.dataframe(logs.head(10)[["timestamp","UPC","Ingredient","Competitor","Result"]], use_container_width=True)

        st.markdown("<div class='section-header'>🗑 Manage Logs</div>", unsafe_allow_html=True)
        if st.button("🗑 Clear Logs"):
            clear_logs_in_db()
            st.success("✅ Logs cleared successfully.")
    else:
        st.info("No logs yet. Run some comparisons to see dashboard data.")

# --- Inventory Page ---
elif menu == "📦 Inventory":
    st.markdown("<div class='main-title'>📦 Inventory Management</div>", unsafe_allow_html=True)
    try:
        medicines = load_inventory_df()
        consumables = load_consumables_df()
        tab1, tab2 = st.tabs(["💊 Medicines", "🛠 Consumables"])

        # Medicines tab
        with tab1:
            st.markdown("<div class='section-header'>💊 Medicines Inventory</div>", unsafe_allow_html=True)
            if medicines is not None and not medicines.empty:
                total_meds = medicines["Ingredient"].nunique()
                total_stock = medicines["Stock"].sum()
            else:
                total_meds = 0
                total_stock = 0
            c1, c2 = st.columns(2)
            c1.metric("💊 Unique Medicines", total_meds)
            c2.metric("📦 Total Stock", int(total_stock))

            st.markdown("<div class='section-header'>➕ Add / Update Medicine</div>", unsafe_allow_html=True)
            with st.form("med_form", clear_on_submit=False):
                a1, a2, a3 = st.columns(3)
                with a1:
                    upc = st.text_input("UPC")
                    ingredient = st.text_input("Ingredient")
                with a2:
                    manufacturer = st.text_input("Manufacturer")
                    batch = st.text_input("Batch")
                with a3:
                    stock = st.number_input("Stock", min_value=0, value=1, step=1)
                    expiry = st.date_input("Expiry Date")
                save_med = st.form_submit_button("💾 Save Medicine")
                if save_med:
                    doc = {
                        "UPC": upc.strip(),
                        "Ingredient": ingredient.strip(),
                        "Manufacturer": manufacturer.strip(),
                        "Batch": batch.strip(),
                        "Stock": int(stock),
                        "Expiry": expiry.isoformat() if expiry else None
                    }
                    save_inventory_item(doc)
                    st.success("Saved medicine.")
                    st.experimental_rerun()

            st.markdown("<div class='section-header'>📋 Medicines List</div>", unsafe_allow_html=True)
            if medicines is None or medicines.empty:
                st.info("No medicines found.")
            else:
                display_meds = medicines.copy()
                if "Expiry" in display_meds.columns:
                    display_meds["Expiry"] = pd.to_datetime(display_meds["Expiry"], errors="coerce").dt.date
                display_meds = display_meds.drop(columns=["_id"], errors="ignore")
                st.dataframe(display_meds, use_container_width=True)

                labels = []
                id_map = []
                for _, row in medicines.iterrows():
                    label = f"{row.get('Ingredient','(no name)')}  |  UPC:{row.get('UPC','')}  |  Batch:{row.get('Batch','')}"
                    labels.append(label)
                    id_map.append(row.get("_id"))
                st.markdown("### Manage a medicine record")
                sel_index = st.selectbox("Select medicine to edit/delete", options=list(range(len(labels))), format_func=lambda i: labels[i]) if labels else None
                if sel_index is not None:
                    selected_id = id_map[sel_index]
                    rec = medicines[medicines["_id"] == selected_id].iloc[0].to_dict()
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        edit_ingredient = st.text_input("Ingredient", value=rec.get("Ingredient",""))
                        edit_upc = st.text_input("UPC", value=rec.get("UPC",""))
                    with m2:
                        edit_manufacturer = st.text_input("Manufacturer", value=rec.get("Manufacturer",""))
                        edit_batch = st.text_input("Batch", value=rec.get("Batch",""))
                    with m3:
                        edit_stock = st.number_input("Stock", min_value=0, value=int(rec.get("Stock",0)))
                        try:
                            edit_expiry = st.date_input("Expiry", value=pd.to_datetime(rec.get("Expiry")).date())
                        except Exception:
                            edit_expiry = st.date_input("Expiry")
                    if st.button("Save changes to selected"):
                        update_fields = {
                            "Ingredient": edit_ingredient.strip(),
                            "UPC": edit_upc.strip(),
                            "Manufacturer": edit_manufacturer.strip(),
                            "Batch": edit_batch.strip(),
                            "Stock": int(edit_stock),
                            "Expiry": edit_expiry.isoformat() if edit_expiry else None
                        }
                        update_inventory_by_id(selected_id, update_fields)
                        st.success("Medicine record updated.")
                        st.experimental_rerun()
                    if st.button("Delete selected medicine"):
                        delete_inventory_by_id(selected_id)
                        st.success("Record deleted.")
                        st.experimental_rerun()

        # Consumables tab
        with tab2:
            st.markdown("<div class='section-header'>🛠 Consumables Inventory</div>", unsafe_allow_html=True)
            if consumables is not None and not consumables.empty:
                total_items = consumables["Item Name"].nunique() if "Item Name" in consumables.columns else len(consumables)
                total_qty = consumables["Quantity in Stock"].fillna(0).astype(int).sum()
            else:
                total_items = 0
                total_qty = 0
            cc1, cc2 = st.columns(2)
            cc1.metric("🛠 Unique Items", total_items)
            cc2.metric("📦 Total Quantity", int(total_qty))

            st.markdown("<div class='section-header'>➕ Add / Update Consumable</div>", unsafe_allow_html=True)
            with st.form("cons_form", clear_on_submit=False):
                x1, x2 = st.columns(2)
                with x1:
                    item_name = st.text_input("Item Name")
                    category = st.text_input("Category")
                    upc_c = st.text_input("UPC")
                with x2:
                    qty = st.number_input("Quantity in Stock", min_value=0, value=1, step=1)
                    expiry_months = st.number_input("Expiry Period (Months)", min_value=0, value=12)
                    storage_temp = st.number_input("Storage Temp (°C)", value=25)
                submit_cons = st.form_submit_button("💾 Save Consumable")
                if submit_cons:
                    doc = {
                        "Item Name": item_name.strip(),
                        "Category": category.strip(),
                        "UPC": upc_c.strip(),
                        "Quantity in Stock": int(qty),
                        "Expiry Period (Months)": int(expiry_months),
                        "Storage Temperature (C)": storage_temp
                    }
                    save_consumable_item(doc)
                    st.success("Saved consumable.")
                    st.experimental_rerun()

            st.markdown("<div class='section-header'>📋 Consumables List</div>", unsafe_allow_html=True)
            if consumables is None or consumables.empty:
                st.info("No consumables found.")
            else:
                display_cons = consumables.copy()
                display_cons = display_cons.drop(columns=["_id"], errors="ignore")
                st.dataframe(display_cons, use_container_width=True)

                clabels = []
                cid_map = []
                for _, row in consumables.iterrows():
                    label = f"{row.get('Item Name','(no name)')}  |  UPC:{row.get('UPC','')}  |  Qty:{int(row.get('Quantity in Stock',0))}"
                    clabels.append(label)
                    cid_map.append(row.get("_id"))
                st.markdown("### Manage a consumable")
                csel_index = st.selectbox("Select consumable to edit/delete", options=list(range(len(clabels))), format_func=lambda i: clabels[i]) if clabels else None
                if csel_index is not None:
                    selected_cid = cid_map[csel_index]
                    crec = consumables[consumables["_id"] == selected_cid].iloc[0].to_dict()
                    e1, e2 = st.columns(2)
                    with e1:
                        new_item = st.text_input("Item Name", value=crec.get("Item Name",""))
                        new_cat = st.text_input("Category", value=crec.get("Category",""))
                        new_qty = st.number_input("Quantity in Stock", min_value=0, value=int(crec.get("Quantity in Stock",0)))
                    with e2:
                        new_upc = st.text_input("UPC", value=crec.get("UPC",""))
                        new_exp_m = st.number_input("Expiry Period (Months)", min_value=0, value=int(crec.get("Expiry Period (Months)",0)))
                        new_safe = st.selectbox("Safe/Not Safe", ["Safe","Not Safe"], index=0 if crec.get("Safe/Not Safe","Safe")=="Safe" else 1)
                    if st.button("Save Consumable Changes"):
                        update_consumable_by_id(selected_cid, {
                            "Item Name": new_item.strip(),
                            "Category": new_cat.strip(),
                            "Quantity in Stock": int(new_qty),
                            "UPC": new_upc.strip(),
                            "Expiry Period (Months)": int(new_exp_m),
                            "Safe/Not Safe": new_safe
                        })
                        st.success("Consumable updated.")
                        st.experimental_rerun()
                    if st.button("Delete Consumable"):
                        delete_consumable_by_id(selected_cid)
                        st.success("Consumable deleted.")
                        st.experimental_rerun()

    except Exception as e:
        st.error(f"⚠ Could not process inventory: {e}")
        st.info("Check your database documents and try again.")

# --- Password Reset Page ---
if menu == "🔑 Change Password":
    password_reset(username)
    st.stop()
