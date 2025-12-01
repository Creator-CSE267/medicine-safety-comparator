# app.py (CLEANED, DB-only, ready to paste)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import plotly.express as px
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
import certifi

# ------------ Login system imports (assumes these modules exist) ------------
from login import login_router
from user_database import init_user_db
from password_reset import password_reset

# ------------ Styling helpers (assumes these modules exist) ------------
from styles import apply_theme, apply_layout_styles, apply_global_css, set_background, show_logo

# --------------- CONFIG ------------------------
SESSION_TIMEOUT_SECONDS = 30 * 60

# --------------- MONGODB CONNECT ----------------

@st.cache_resource
def get_db():
    """
    Robust MongoDB connect for Streamlit Cloud.
    Uses certifi CA bundle for TLS verification.
    """
    try:
        uri = st.secrets["MONGO"]["URI"]
        dbname = st.secrets["MONGO"]["DBNAME"]
    except Exception:
        uri = os.getenv("MONGO_URI")
        dbname = os.getenv("MONGO_DBNAME")

    if not uri or not dbname:
        st.error("MongoDB configuration missing. Add MONGO.URI and MONGO.DBNAME to Streamlit secrets or environment.")
        st.stop()

    client_opts = {
        "serverSelectionTimeoutMS": 20000,
        "connectTimeoutMS": 20000,
        "tls": True,
        "tlsCAFile": certifi.where(),
    }

    try:
        client = MongoClient(uri, **client_opts)
        client.admin.command("ping")
    except Exception as e:
        st.error(f"Could not connect to MongoDB: {e}")
        raise

    return client[dbname]

# Initialize DB and collection handles immediately
db = get_db()
try:
    inv_col = db["inventory"]         # inventory UI collection
    consumables_col = db["consumables"]
    med_coll = db["medicines"]        # ML dataset collection (used by model)
    logs_col = db["usage_log"]
except Exception as e:
    st.error(f"MongoDB collections could not be initialized: {e}")
    raise

# -----------------------------
# Minimal MongoDB helper functions (inventory / consumables / logs)
# -----------------------------
def load_inventory_df():
    """Return inventory as a DataFrame with _id as string (safe for display)."""
    try:
        docs = list(inv_col.find({}))
    except Exception as e:
        st.error(f"Failed to load inventory: {e}")
        return pd.DataFrame(columns=["UPC", "Ingredient", "Manufacturer", "Batch", "Stock", "Expiry", "_id"])
    for d in docs:
        d["_id"] = str(d["_id"])
    df = pd.DataFrame(docs)
    for c in ["UPC", "Ingredient", "Manufacturer", "Batch", "Stock", "Expiry", "_id"]:
        if c not in df.columns:
            df[c] = None
    if "Stock" in df.columns:
        df["Stock"] = pd.to_numeric(df["Stock"], errors="coerce").fillna(0).astype(int)
    return df

def save_inventory_doc(doc: dict):
    """Insert or update inventory item by UPC+Batch."""
    key = {"UPC": doc.get("UPC",""), "Batch": doc.get("Batch","")}
    if not key["UPC"] and not key["Batch"]:
        inv_col.insert_one(doc)
        return
    existing = inv_col.find_one(key)
    if existing:
        inv_col.update_one({"_id": existing["_id"]}, {"$set": doc})
    else:
        inv_col.insert_one(doc)

def delete_inventory(id_str: str):
    try:
        inv_col.delete_one({"_id": ObjectId(id_str)})
    except Exception as e:
        st.error(f"Delete failed: {e}")

def load_consumables_df():
    try:
        docs = list(consumables_col.find({}))
    except Exception as e:
        st.error(f"Failed to load consumables: {e}")
        return pd.DataFrame(columns=[
            "Item Name","Category","Material Type","Sterility Level",
            "Expiry Period (Months)","Storage Temperature (C)","Quantity in Stock",
            "Usage Type","Certification Standard","UPC","Safe/Not Safe","_id"
        ])
    for d in docs:
        d["_id"] = str(d["_id"])
    df = pd.DataFrame(docs)
    if "Quantity in Stock" in df.columns:
        df["Quantity in Stock"] = pd.to_numeric(df["Quantity in Stock"], errors="coerce").fillna(0).astype(int)
    return df

def save_consumable_doc(doc: dict):
    key = {"UPC": doc.get("UPC")} if doc.get("UPC") else None
    existing = consumables_col.find_one(key) if key else None
    if existing:
        consumables_col.update_one({"_id": existing["_id"]}, {"$set": doc})
    else:
        consumables_col.insert_one(doc)

def delete_consumable(id_str: str):
    try:
        consumables_col.delete_one({"_id": ObjectId(id_str)})
    except Exception as e:
        st.error(f"Delete failed: {e}")

def append_log(entry: dict):
    if "timestamp" in entry:
        try:
            entry["timestamp"] = pd.to_datetime(entry["timestamp"]).isoformat()
        except:
            entry["timestamp"] = datetime.now().isoformat()
    else:
        entry["timestamp"] = datetime.now().isoformat()
    logs_col.insert_one(entry)

def load_logs_df(limit=5000):
    try:
        docs = list(logs_col.find({}).sort([("_id", -1)]).limit(limit))
    except Exception as e:
        st.error(f"Failed to load logs: {e}")
        return pd.DataFrame(columns=["timestamp", "UPC", "Ingredient", "Competitor", "Result"])
    for d in docs:
        d["_id"] = str(d["_id"])
        if "timestamp" in d:
            d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    return pd.DataFrame(docs)

def clear_logs_in_db():
    logs_col.delete_many({})

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

# ------- AUTH SUCCESS → Read user data BEFORE sidebar ----------
username = st.session_state.get("username", "User")
role = st.session_state.get("role", "user")
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

# -------------------- Sidebar with avatar + logout + role menu --------------------
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

    # Role-based navigation
    if role == "admin":
        menu = st.sidebar.radio("📌 Navigation", ["📊 Dashboard", "📦 Inventory", "🔑 Change Password"])
    elif role == "pharmacist":
        menu = st.sidebar.radio("📌 Navigation", ["🧪 Testing", "📦 Inventory", "🔑 Change Password"])
    else:
        menu = st.sidebar.radio("📌 Navigation", ["📦 Inventory"])

    st.sidebar.markdown("---")
    st.sidebar.write("© 2025 MedSafe AI")

# ===============================
# Load Medicine Dataset for ML (from MongoDB med_coll)
# ===============================
try:
    docs = list(med_coll.find({}))
except Exception as e:
    st.error(f"Could not read 'medicines' collection: {e}")
    st.stop()

if not docs:
    st.error("❌ No records found in MongoDB 'medicines' collection. Add data first.")
    st.stop()

df = pd.DataFrame(docs)

# Drop internal mongo _id if present
if "_id" in df.columns:
    df = df.drop(columns=["_id"])

# Ensure required columns exist
required_cols = [
    "UPC", "Active Ingredient", "Disease/Use Case",
    "Days Until Expiry", "Storage Temperature (C)",
    "Dissolution Rate (%)", "Disintegration Time (minutes)",
    "Impurity Level (%)", "Assay Purity (%)",
    "Warning Labels Present", "Safe/Not Safe"
]
for col in required_cols:
    if col not in df.columns:
        df[col] = None

# Clean values
df["UPC"] = df["UPC"].astype(str).str.strip()
df["Active Ingredient"] = df["Active Ingredient"].fillna("Unknown")
df["Disease/Use Case"] = df["Disease/Use Case"].fillna("Unknown")
df["Safe/Not Safe"] = df["Safe/Not Safe"].fillna("Safe")

# Convert Warning Labels Present (Yes/No) to numeric 1/0; otherwise coerce numeric
if df["Warning Labels Present"].dtype == "object":
    df["Warning Labels Present"] = df["Warning Labels Present"].map({"Yes": 1, "No": 0}).fillna(0)
else:
    df["Warning Labels Present"] = pd.to_numeric(df["Warning Labels Present"], errors="coerce").fillna(0)

# Drop rows missing crucial fields for ML
df = df.dropna(subset=["Active Ingredient", "Safe/Not Safe"])

# Encode target
y = df["Safe/Not Safe"].astype(str)
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Numeric columns for ML - ensure they exist and are numeric
numeric_cols = [
    "Days Until Expiry",
    "Storage Temperature (C)",
    "Dissolution Rate (%)",
    "Disintegration Time (minutes)",
    "Impurity Level (%)",
    "Assay Purity (%)",
    "Warning Labels Present"
]
for c in numeric_cols:
    if c not in df.columns:
        df[c] = np.nan
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Final ML input
X = df[["Active Ingredient", "Disease/Use Case"] + numeric_cols]

# Consistency check to avoid sklearn ValueError
if len(X) != len(y_enc):
    st.error(f"ML data mismatch: X rows={len(X)} y rows={len(y_enc)}. Check 'medicines' docs.")
    st.stop()

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

model = train_model(X, y_enc)

# ===============================
# Safety Rules & suggestions
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
        if val is None:
            continue
        try:
            val_f = float(val)
        except:
            val_f = val
        if "min" in rule and val_f < rule["min"]:
            suggestions.append(f"Increase *{col}* (min {rule['min']}).")
        if "max" in rule and val_f > rule["max"]:
            suggestions.append(f"Reduce *{col}* (max {rule['max']}).")
        if "range" in rule:
            low, high = rule["range"]
            if not (low <= val_f <= high):
                suggestions.append(f"Keep *{col}* within {low}-{high}.")
    return suggestions

# ===============================
# Pages & Navigation
# ===============================

# ------------------------------
# Testing Page (pharmacist + maybe admin)
# ------------------------------
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
            st.error("❌ UPC not found in medicines collection.")
    elif ingredient_input:
        match = df[df["Active Ingredient"].str.lower() == ingredient_input.lower()]
        if not match.empty:
            selected_row = match.iloc[0]
            upc_input = selected_row["UPC"]
            st.success(f"✅ Ingredient found → UPC: {upc_input}")
        else:
            st.error("❌ Ingredient not found in medicines collection.")

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
            ax.bar(x - width/2, base_values, width, label="Standard Medicine")
            ax.bar(x + width/2, comp_values, width, label="Competitor Medicine")
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

            # PDF report (optional)
            try:
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
                from reportlab.lib.styles import getSampleStyleSheet
                from reportlab.lib.pagesizes import A4

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

                if result.lower() == "not safe" and suggestions:
                    elements.append(Paragraph("<b>⚠ Suggested Improvements:</b>", styles["Heading2"]))
                    for s in suggestions:
                        elements.append(Paragraph(f"- {s}", styles["Normal"]))
                    elements.append(Spacer(1, 12))

                chart_buffer = io.BytesIO()
                fig.savefig(chart_buffer, format="png")
                chart_buffer.seek(0)
                elements.append(RLImage(chart_buffer, width=400, height=250))
                elements.append(Spacer(1, 12))

                doc.build(elements)
                buffer.seek(0)

                st.download_button(
                    label="⬇ Download PDF Report",
                    data=buffer,
                    file_name=f"Medicine_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
            except Exception:
                # PDF optional; ignore errors silently
                pass

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
        fig_trend.update_layout(title_x=0.5)
        st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown("<div class='section-header'>📋 Recent Activity</div>", unsafe_allow_html=True)
        st.dataframe(logs.head(10)[["timestamp", "UPC", "Ingredient", "Competitor", "Result"]], use_container_width=True)

        st.markdown("<div class='section-header'>🗑 Manage Logs</div>", unsafe_allow_html=True)
        if st.button("🗑 Clear Logs"):
            clear_logs_in_db()
            st.success("✅ Logs cleared successfully.")
    else:
        st.info("No logs yet. Run some comparisons to see dashboard data.")

# =========================================================
# 📦 INVENTORY PAGE (separated tabs for medicines & consumables)
# =========================================================
elif menu == "📦 Inventory":
    st.header("📦 Inventory Management")

    tab1, tab2 = st.tabs(["💊 Medicines", "🛠 Consumables"])

    # -------------------------
    # Medicines Tab (inv_col)
    # -------------------------
    with tab1:
        inv = load_inventory_df()
        total_items = len(inv)
        total_stock = inv["Stock"].fillna(0).astype(int).sum() if not inv.empty and "Stock" in inv else 0
        c1, c2 = st.columns(2)
        c1.metric("Total Items", total_items)
        c2.metric("Total Stock", int(total_stock))

        st.subheader("➕ Add / Update Medicine")
        with st.form("med_form"):
            colA, colB, colC = st.columns(3)
            upc = colA.text_input("UPC")
            ingredient = colA.text_input("Ingredient")
            mf = colB.text_input("Manufacturer")
            batch = colB.text_input("Batch")
            stock = colC.number_input("Stock", min_value=0, value=1, step=1)
            expiry = colC.date_input("Expiry Date")
            if st.form_submit_button("Save"):
                doc = {
                    "UPC": upc.strip(),
                    "Ingredient": ingredient.strip(),
                    "Manufacturer": mf.strip(),
                    "Batch": batch.strip(),
                    "Stock": int(stock),
                    "Expiry": expiry.isoformat(),
                }
                existing = inv_col.find_one({"UPC": upc.strip(), "Batch": batch.strip()})
                if existing:
                    inv_col.update_one({"_id": existing["_id"]}, {"$set": doc})
                else:
                    inv_col.insert_one(doc)
                st.success("Saved successfully.")
                st.experimental_rerun()

        st.subheader("📋 Inventory List")
        if inv.empty:
            st.info("No inventory records found.")
        else:
            show = inv.drop(columns=["_id"], errors="ignore")
            if "Expiry" in show:
                show["Expiry"] = pd.to_datetime(show["Expiry"], errors="coerce").dt.date
            st.dataframe(show, use_container_width=True)

            labels = [
                f"{row['Ingredient']} | UPC:{row['UPC']} | Batch:{row['Batch']}"
                for _, row in inv.iterrows()
            ]
            selected_index = st.selectbox("Select item", list(range(len(labels))), format_func=lambda i: labels[i])
            selected_id = inv.iloc[selected_index]["_id"]
            record = inv.iloc[selected_index]

            col1, col2, col3 = st.columns(3)
            new_ing = col1.text_input("Ingredient", value=record.get("Ingredient",""))
            new_upc = col1.text_input("UPC", value=record.get("UPC",""))
            new_mf = col2.text_input("Manufacturer", value=record.get("Manufacturer",""))
            new_batch = col2.text_input("Batch", value=record.get("Batch",""))
            new_stock = col3.number_input("Stock", min_value=0, value=int(record.get("Stock",0)))
            try:
                new_exp = col3.date_input("Expiry", value=pd.to_datetime(record.get("Expiry")).date())
            except:
                new_exp = col3.date_input("Expiry")

            if st.button("Save Changes"):
                inv_col.update_one({"_id": ObjectId(selected_id)}, {
                    "$set": {
                        "Ingredient": new_ing.strip(),
                        "UPC": new_upc.strip(),
                        "Manufacturer": new_mf.strip(),
                        "Batch": new_batch.strip(),
                        "Stock": int(new_stock),
                        "Expiry": new_exp.isoformat(),
                    }
                })
                st.success("Updated successfully.")
                st.experimental_rerun()

            if st.button("Delete Item"):
                inv_col.delete_one({"_id": ObjectId(selected_id)})
                st.success("Deleted.")
                st.experimental_rerun()

    # -------------------------
    # Consumables Tab (consumables_col)
    # -------------------------
    with tab2:
        cons = load_consumables_df()
        total_cons = cons["Item Name"].nunique() if not cons.empty and "Item Name" in cons else 0
        total_qty = cons["Quantity in Stock"].fillna(0).astype(int).sum() if not cons.empty and "Quantity in Stock" in cons else 0
        cc1, cc2 = st.columns(2)
        cc1.metric("🛠 Unique Items", total_cons)
        cc2.metric("📦 Total Quantity", int(total_qty))

        st.subheader("➕ Add / Update Consumable")
        with st.form("cons_form"):
            x1, x2 = st.columns(2)
            item_name = x1.text_input("Item Name")
            category = x1.text_input("Category")
            upc_c = x2.text_input("UPC")
            qty = x2.number_input("Quantity in Stock", min_value=0, value=1, step=1)
            expiry_months = x2.number_input("Expiry Period (Months)", min_value=0, value=12)
            storage_temp = x2.number_input("Storage Temp (°C)", value=25)
            if st.form_submit_button("Save Consumable"):
                doc = {
                    "Item Name": item_name.strip(),
                    "Category": category.strip(),
                    "UPC": upc_c.strip(),
                    "Quantity in Stock": int(qty),
                    "Expiry Period (Months)": int(expiry_months),
                    "Storage Temperature (C)": storage_temp
                }
                save_consumable_doc(doc)
                st.success("Saved consumable to MongoDB")
                st.experimental_rerun()

        st.subheader("📋 Consumables List")
        if cons.empty:
            st.info("No consumables found.")
        else:
            display_cons = cons.copy().drop(columns=["_id"], errors="ignore")
            st.dataframe(display_cons, use_container_width=True)

            clabels = []
            cid_map = []
            for _, row in cons.iterrows():
                label = f"{row.get('Item Name','(no name)')}  |  UPC:{row.get('UPC','')}  |  Qty:{int(row.get('Quantity in Stock',0))}"
                clabels.append(label)
                cid_map.append(row.get("_id"))

            st.markdown("### Manage a consumable")
            csel_index = st.selectbox("Select consumable to edit/delete", options=list(range(len(clabels))), format_func=lambda i: clabels[i]) if clabels else None

            if csel_index is not None:
                selected_cid = cid_map[csel_index]
                crec = cons[cons["_id"] == selected_cid].iloc[0].to_dict()

                e1, e2 = st.columns(2)
                new_item = e1.text_input("Item Name", value=crec.get("Item Name", ""))
                new_cat = e1.text_input("Category", value=crec.get("Category", ""))
                new_qty = e2.number_input("Quantity in Stock", min_value=0, value=int(crec.get("Quantity in Stock", 0)))
                new_upc = e2.text_input("UPC", value=crec.get("UPC", ""))
                new_exp_m = e2.number_input("Expiry Period (Months)", min_value=0, value=int(crec.get("Expiry Period (Months)", 0)))
                new_safe = e2.selectbox("Safe/Not Safe", ["Safe", "Not Safe"], index=0 if crec.get("Safe/Not Safe","Safe")=="Safe" else 1)

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
                    consumables_col.delete_one({"_id": ObjectId(selected_cid)})
                    st.success("Consumable deleted.")
                    st.experimental_rerun()

# ===============================
# STEP — PASSWORD RESET PAGE
# ===============================
if menu == "🔑 Change Password":
    password_reset(username)
    st.stop()
