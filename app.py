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
from datetime import datetime, timedelta
from PIL import Image
import io
from pymongo import MongoClient
from bson import ObjectId
import os as _os
import certifi
from pymongo import MongoClient
import pandas as pd
import os
from bson import ObjectId

# ------------ Login system imports ------------
from login import login_router
from user_database import init_user_db
from password_reset import password_reset

# ------------ Styling helpers -----------------
from styles import apply_theme, apply_layout_styles, apply_global_css, set_background, show_logo

# --------------- CONFIG ------------------------
SESSION_TIMEOUT_SECONDS = 30 * 60

# --------------- MONGODB CONNECT ----------------

@st.cache_resource
def get_db():
    """
    Robust MongoDB connect for Streamlit Cloud.
    Uses certifi CA bundle for TLS verification (fixes SSL handshake errors).
    """


    # Load values from Streamlit Secrets
    try:
        uri = st.secrets["MONGO"]["URI"]
        dbname = st.secrets["MONGO"]["DBNAME"]
    except Exception:
        uri = _os.getenv("MONGO_URI")
        dbname = _os.getenv("MONGO_DBNAME")

    if not uri or not dbname:
        st.error("MongoDB configuration missing. Add MONGO.URI and MONGO.DBNAME to Streamlit secrets.")
        st.stop()

    client_opts = {
        "serverSelectionTimeoutMS": 20000,
        "connectTimeoutMS": 20000,
        "tls": True,
        "tlsCAFile": certifi.where(),   # <- this is the main fix
    }

    try:
        client = MongoClient(uri, **client_opts)
        client.admin.command("ping")   # force handshake now (fail early)
    except Exception as e:
        st.error(f"Could not connect to MongoDB: {e}")
        raise

    return client[dbname]

# ---------------------------
# SAFE ONE-CLICK CSV → Mongo MIGRATION
# Paste this anywhere after your get_db() function (but it's safe to paste below the db = get_db() lines too)
# ---------------------------

def migrate_csv_to_mongo_safe():
    """
    Robust migration that obtains DB/collections inside the function so it never
    fails with NameError if called earlier in file execution.
    """
    # get DB & collections (works whether db globals exist or not)
    try:
        db_local = get_db()
    except Exception:
        # fallback: if get_db is not available, try to read globals
        db_local = globals().get("db")
        if db_local is None:
            raise RuntimeError("Database not available (get_db failed and global db is not set).")

    coll = db_local["inventory"]
    cons_coll = db_local["consumables"]

    # ---------- migrate inventory.csv ----------
    inv_path = "inventory.csv"
    if os.path.exists(inv_path):
        df = pd.read_csv(inv_path)
        for _, r in df.iterrows():
            doc = {
                "UPC": str(r.get("UPC", "")).strip(),
                "Ingredient": str(r.get("Ingredient", r.get("Active Ingredient", ""))).strip(),
                "Manufacturer": str(r.get("Manufacturer", "")).strip(),
                "Batch": str(r.get("Batch", r.get("Batch Number", ""))).strip(),
                "Stock": int(r.get("Stock", r.get("Quantity", 0)) or 0),
                "Expiry": str(r.get("Expiry", "")) if pd.notnull(r.get("Expiry", None)) else None
            }
            # upsert by UPC+Batch
            key = {"UPC": doc["UPC"], "Batch": doc["Batch"]}
            if doc["UPC"] == "" and doc["Batch"] == "":
                # if both empty, insert as new doc (avoid overwriting)
                coll.insert_one(doc)
            else:
                existing = coll.find_one(key)
                if existing:
                    coll.update_one({"_id": existing["_id"]}, {"$set": doc})
                else:
                    coll.insert_one(doc)

    # ---------- migrate consumables_dataset.csv ----------
    cons_path = "consumables_dataset.csv"
    if os.path.exists(cons_path):
        dfc = pd.read_csv(cons_path)
        for _, r in dfc.iterrows():
            doc = {
                "Item Name": str(r.get("Item Name", "")).strip(),
                "Category": str(r.get("Category", "")).strip(),
                "Material Type": str(r.get("Material Type", "")).strip(),
                "Sterility Level": str(r.get("Sterility Level", "")).strip(),
                "Expiry Period (Months)": int(r.get("Expiry Period (Months)", 0) or 0),
                "Storage Temperature (C)": r.get("Storage Temperature (C)", None),
                "Quantity in Stock": int(r.get("Quantity in Stock", r.get("Quantity", 0)) or 0),
                "Usage Type": str(r.get("Usage Type", "")).strip(),
                "Certification Standard": str(r.get("Certification Standard", "")).strip(),
                "UPC": str(r.get("UPC", "")).strip(),
                "Safe/Not Safe": str(r.get("Safe/Not Safe", "Safe")).strip()
            }
            key = {"UPC": doc["UPC"]} if doc["UPC"] else None
            if key:
                existing = cons_coll.find_one(key)
            else:
                existing = None

            if existing:
                cons_coll.update_one({"_id": existing["_id"]}, {"$set": doc})
            else:
                cons_coll.insert_one(doc)

    return True


# BUTTON (paste this where you want the button — sidebar recommended)
if st.sidebar.button("📤 MIGRATE ALL CSV → MONGO (safe)"):
    with st.spinner("Migrating CSV data to MongoDB..."):
        try:
            migrate_csv_to_mongo_safe()
            st.success("✅ Migration complete! Refresh the app.")
        except Exception as e:
            st.error(f"Migration failed: {e}")
            raise



db = get_db()
collection = db["inventory"]

# Additional collections
consumables_col = db["consumables"]
logs_col = db["usage_log"]

# ------------ INIT DB (users) -----------------------
init_user_db()

# --------------- SESSION DEFAULTS --------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None
if "role" not in st.session_state:
    st.session_state["role"] = None
if "last_active" not in st.session_state:
    st.session_state["last_active"] = None


# ---------------- TIMEOUT CHECK ----------------
def session_is_timed_out():
    last = st.session_state.get("last_active")
    if not last:
        return False
    return (datetime.now() - datetime.fromisoformat(last)).total_seconds() > SESSION_TIMEOUT_SECONDS


if st.session_state["authenticated"] and session_is_timed_out():
    st.warning("Session timed out. Login again.")
    st.session_state["authenticated"] = False
    st.rerun()


# ---------------- LOGIN FIRST -------------------
if not st.session_state["authenticated"]:
    login_router()
    st.stop()


# ------- AUTH SUCCESS → READ USER DATA ----------
username = st.session_state["username"]
role = st.session_state["role"]
st.session_state["last_active"] = datetime.now().isoformat()


# ------------- APPLY THEME AFTER LOGIN ----------
st.set_page_config(page_title="Medicine Safety Comparator",
                   page_icon="💊",
                   layout="wide")

apply_theme()
apply_layout_styles()
apply_global_css()

set_background("bg1.jpg")
show_logo("logo.png")

st.title("💊 Medicine Safety Comparator")

# ===============================
# File Paths (kept for compatibility but CSVs are not required now)
# ===============================
MEDICINE_FILE = "medicine_dataset.csv"
INVENTORY_FILE = "inventory.csv"
CONSUMABLES_FILE = "consumables_dataset.csv"
LOG_FILE = "usage_log.csv"  # legacy name (we now use logs_col)

# ===============================
# DB helper functions (Inventory / Consumables / Logs)
# ===============================
def _ensure_columns(df, expected):
    for c in expected:
        if c not in df.columns:
            df[c] = None
    return df

# --- Inventory helpers ---
def load_medicines_df():
    docs = list(collection.find({}))
    if not docs:
        cols = ["UPC", "Ingredient", "Manufacturer", "Batch", "Stock", "Expiry"]
        return pd.DataFrame(columns=cols)
    for d in docs:
        d["_id"] = str(d["_id"])
    df = pd.DataFrame(docs)
    expected = ["UPC", "Ingredient", "Manufacturer", "Batch", "Stock", "Expiry"]
    df = _ensure_columns(df, expected)
    if "Expiry" in df.columns:
        df["Expiry"] = pd.to_datetime(df["Expiry"], errors="coerce")
    # normalize types
    if "Stock" in df.columns:
        df["Stock"] = pd.to_numeric(df["Stock"], errors="coerce").fillna(0).astype(int)
    return df

def save_medicine_to_db(doc):
    # doc: dict with UPC, Ingredient, Manufacturer, Batch, Stock, Expiry (expiry may be date or iso str)
    q = {"UPC": doc.get("UPC"), "Batch": doc.get("Batch")}
    existing = collection.find_one(q)
    # Normalize expiry
    if "Expiry" in doc and pd.notnull(doc["Expiry"]):
        try:
            doc["Expiry"] = pd.to_datetime(doc["Expiry"]).isoformat()
        except Exception:
            doc["Expiry"] = str(doc["Expiry"])
    if existing:
        upd = {}
        if "Stock" in doc:
            try:
                upd["Stock"] = int(existing.get("Stock", 0)) + int(doc.get("Stock", 0))
            except:
                upd["Stock"] = doc.get("Stock")
        if "Expiry" in doc:
            upd["Expiry"] = doc["Expiry"]
        if upd:
            collection.update_one({"_id": existing["_id"]}, {"$set": upd})
        return str(existing["_id"])
    else:
        collection.insert_one(doc)
        return None

def delete_medicine_by_id(obj_id):
    try:
        collection.delete_one({"_id": ObjectId(obj_id)})
    except Exception as e:
        st.error(f"Delete failed: {e}")

# --- Consumables helpers ---
def load_consumables_df():
    docs = list(consumables_col.find({}))
    if not docs:
        cols = [
            "Item Name", "Category", "Material Type", "Sterility Level",
            "Expiry Period (Months)", "Storage Temperature (C)", "Quantity in Stock",
            "Usage Type", "Certification Standard", "UPC", "Safe/Not Safe"
        ]
        return pd.DataFrame(columns=cols)
    for d in docs:
        d["_id"] = str(d["_id"])
    df = pd.DataFrame(docs)
    # normalize numeric
    if "Quantity in Stock" in df.columns:
        df["Quantity in Stock"] = pd.to_numeric(df["Quantity in Stock"], errors="coerce").fillna(0).astype(int)
    return df

def save_consumable_to_db(doc):
    # doc should include keys matching the consumables schema
    q = {"UPC": doc.get("UPC")} if doc.get("UPC") else None
    existing = consumables_col.find_one(q) if q else None
    if existing:
        upd = {}
        if "Quantity in Stock" in doc:
            try:
                upd["Quantity in Stock"] = int(existing.get("Quantity in Stock", 0)) + int(doc.get("Quantity in Stock", 0))
            except:
                upd["Quantity in Stock"] = doc.get("Quantity in Stock")
        if "Expiry Period (Months)" in doc:
            upd["Expiry Period (Months)"] = doc.get("Expiry Period (Months)")
        if upd:
            consumables_col.update_one({"_id": existing["_id"]}, {"$set": upd})
        return str(existing["_id"])
    else:
        consumables_col.insert_one(doc)
        return None

# --- Logs helpers ---
def append_log(entry: dict):
    # ensure timestamp stored as ISO string
    if "timestamp" in entry:
        try:
            entry["timestamp"] = pd.to_datetime(entry["timestamp"]).isoformat()
        except:
            entry["timestamp"] = datetime.now().isoformat()
    else:
        entry["timestamp"] = datetime.now().isoformat()
    logs_col.insert_one(entry)

def load_logs_df(limit=5000):
    docs = list(logs_col.find({}).sort([("_id", -1)]).limit(limit))
    if not docs:
        cols = ["timestamp", "UPC", "Ingredient", "Competitor", "Result"]
        return pd.DataFrame(columns=cols)
    for d in docs:
        d["_id"] = str(d["_id"])
        if "timestamp" in d:
            d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    df = pd.DataFrame(docs)
    return df

def clear_logs_in_db():
    logs_col.delete_many({})


# ===============================
# Load Medicine Dataset (local file for ML model)
# ===============================
if not os.path.exists(MEDICINE_FILE):
    st.error(f"Required dataset '{MEDICINE_FILE}' not found in repository.")
    st.stop()

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
# Pages & Navigation
# ===============================

# --------------------
# Sidebar with avatar + logout + role menu
# --------------------
def render_avatar(username, size=72):
    avatar_path_png = os.path.join("avatars", f"{username}.png")
    avatar_path_jpg = os.path.join("avatars", f"{username}.jpg")
    if os.path.exists(avatar_path_png):
        st.sidebar.image(avatar_path_png, width=size)
        return
    if os.path.exists(avatar_path_jpg):
        st.sidebar.image(avatar_path_jpg, width=size)
        return

    initials = "".join([p[0] for p in username.split()][:2]).upper() if username else "U"
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
        st.success("Logged out. Redirecting to login...")
        st.rerun()

    if role == "admin":
        menu = st.sidebar.radio("📌 Navigation", ["📊 Dashboard", "📦 Inventory", "🔑 Change Password"])
    elif role == "pharmacist":
        menu = st.sidebar.radio("📌 Navigation", ["🧪 Testing", "📦 Inventory", "🔑 Change Password"])
    else:
        menu = st.sidebar.radio("📌 Navigation", ["📦 Inventory"])

    st.sidebar.markdown("---")
    st.sidebar.write("© 2025 MedSafe AI")


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

            # Log (save to Mongo)
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "UPC": upc_input,
                "Ingredient": ingredient_input,
                "Competitor": comp_name,
                "Result": result
            }
            append_log(log_entry)

            # --- PDF Report Download ---
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.pagesizes import A4

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
    apply_global_css()
    st.markdown("<div class='main-title'>📊 Medicine Safety Analytics Dashboard</div>", unsafe_allow_html=True)

    # Load logs from DB
    logs = load_logs_df(limit=5000)

    if not logs.empty:
        # KPI Cards
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

        # Trend Over Time
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

        # Recent Logs
        st.markdown("<div class='section-header'>📋 Recent Activity</div>", unsafe_allow_html=True)
        st.dataframe(
            logs.head(10)[["timestamp", "UPC", "Ingredient", "Competitor", "Result"]],
            use_container_width=True
        )

        # Clear Logs Button
        st.markdown("<div class='section-header'>🗑 Manage Logs</div>", unsafe_allow_html=True)
        if st.button("🗑 Clear Logs"):
            clear_logs_in_db()
            st.success("✅ Logs cleared successfully.")
    else:
        st.info("No logs yet. Run some comparisons to see dashboard data.")


# --- 📦 Inventory Page ---
# --- 📦 Inventory Page (clean UI: hide _id but use it internally) ---
elif menu == "📦 Inventory":
    st.markdown("<div class='main-title'>📦 Inventory Management (MongoDB)</div>", unsafe_allow_html=True)

    try:
        # Load collections (each returns a DataFrame with _id as string)
        medicines = load_medicines()  # expects _id column as string
        consumables = load_consumables()

        tab1, tab2 = st.tabs(["💊 Medicines", "🛠 Consumables"])

        # -------------------------
        # 💊 Medicines Tab (clean)
        # -------------------------
        with tab1:
            st.markdown("<div class='section-header'>💊 Medicines Inventory</div>", unsafe_allow_html=True)

            # KPIs
            if medicines is None or medicines.empty:
                total_meds = 0
                total_stock = 0
            else:
                total_meds = medicines["Ingredient"].nunique()
                total_stock = medicines["Stock"].fillna(0).astype(int).sum()

            c1, c2 = st.columns(2)
            c1.metric("💊 Unique Medicines", total_meds)
            c2.metric("📦 Total Stock", int(total_stock))

            # --- Add / Update Medicine (form) ---
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
                    save_medicine(doc)
                    st.success("Saved medicine to MongoDB")
                    st.experimental_rerun()

            st.markdown("<div class='section-header'>📋 Medicines List</div>", unsafe_allow_html=True)

            # Prepare display dataframe (hide _id)
            if medicines is None or medicines.empty:
                st.info("No medicines found.")
            else:
                display_meds = medicines.copy()
                # Convert expiry to readable date
                if "Expiry" in display_meds.columns:
                    display_meds["Expiry"] = pd.to_datetime(display_meds["Expiry"], errors="coerce").dt.date
                # Hide internal _id from display
                display_meds = display_meds.drop(columns=["_id"], errors="ignore")
                st.dataframe(display_meds, use_container_width=True)

                # Build selection list for manage area (human readable labels -> map to _id)
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
                        edit_ingredient = st.text_input("Ingredient", value=rec.get("Ingredient", ""))
                        edit_upc = st.text_input("UPC", value=rec.get("UPC", ""))
                    with m2:
                        edit_manufacturer = st.text_input("Manufacturer", value=rec.get("Manufacturer", ""))
                        edit_batch = st.text_input("Batch", value=rec.get("Batch", ""))
                    with m3:
                        edit_stock = st.number_input("Stock", min_value=0, value=int(rec.get("Stock", 0)))
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
                        # update by ObjectId string
                        collection.update_one({"_id": ObjectId(selected_id)}, {"$set": update_fields})
                        st.success("Medicine record updated.")
                        st.experimental_rerun()

                    if st.button("Delete selected medicine"):
                        delete_medicine(selected_id)
                        st.success("Record deleted.")
                        st.experimental_rerun()

        # -------------------------
        # 🛠 Consumables Tab (clean)
        # -------------------------
        with tab2:
            st.markdown("<div class='section-header'>🛠 Consumables Inventory</div>", unsafe_allow_html=True)

            # KPIs
            if consumables is None or consumables.empty:
                total_cons = 0
                total_qty = 0
            else:
                total_cons = consumables["Item Name"].nunique() if "Item Name" in consumables.columns else len(consumables)
                total_qty = consumables["Quantity in Stock"].fillna(0).astype(int).sum()

            cc1, cc2 = st.columns(2)
            cc1.metric("🛠 Unique Items", total_cons)
            cc2.metric("📦 Total Quantity", int(total_qty))

            # Add / Update consumable
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
                    save_consumable(doc)
                    st.success("Saved consumable to MongoDB")
                    st.experimental_rerun()

            st.markdown("<div class='section-header'>📋 Consumables List</div>", unsafe_allow_html=True)
            if consumables is None or consumables.empty:
                st.info("No consumables found.")
            else:
                display_cons = consumables.copy()
                display_cons = display_cons.drop(columns=["_id"], errors="ignore")
                st.dataframe(display_cons, use_container_width=True)

                # selection list for consumables
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
                        new_item = st.text_input("Item Name", value=crec.get("Item Name", ""))
                        new_cat = st.text_input("Category", value=crec.get("Category", ""))
                        new_qty = st.number_input("Quantity in Stock", min_value=0, value=int(crec.get("Quantity in Stock", 0)))
                    with e2:
                        new_upc = st.text_input("UPC", value=crec.get("UPC", ""))
                        new_exp_m = st.number_input("Expiry Period (Months)", min_value=0, value=int(crec.get("Expiry Period (Months)", 0)))
                        new_safe = st.selectbox("Safe/Not Safe", ["Safe", "Not Safe"], index=0 if crec.get("Safe/Not Safe","Safe")=="Safe" else 1)

                    if st.button("Save Consumable Changes"):
                        consumables_col.update_one({"_id": ObjectId(selected_cid)}, {"$set": {
                            "Item Name": new_item.strip(),
                            "Category": new_cat.strip(),
                            "Quantity in Stock": int(new_qty),
                            "UPC": new_upc.strip(),
                            "Expiry Period (Months)": int(new_exp_m),
                            "Safe/Not Safe": new_safe
                        }})
                        st.success("Consumable updated.")
                        st.experimental_rerun()

                    if st.button("Delete Consumable"):
                        delete_consumable(selected_cid)
                        st.success("Consumable deleted.")
                        st.experimental_rerun()

    except Exception as e:
        st.error(f"⚠ Could not process inventory: {e}")
        st.info("Try fixing the database documents if the issue persists.")


# ===============================
# STEP 6 — PASSWORD RESET PAGE
# ===============================

if menu == "🔑 Change Password":
    password_reset(username)
    st.stop()
