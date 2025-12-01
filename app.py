
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
from pymongo import MongoClient
from bson import ObjectId
import certifi
import os

# Login + Theme
from login import login_router
from user_database import init_user_db
from password_reset import password_reset
from styles import apply_theme, apply_layout_styles, apply_global_css, set_background, show_logo

SESSION_TIMEOUT_SECONDS = 1800  # 30 min

# --------------------------
# MongoDB Connection (robust)
# --------------------------
@st.cache_resource
def get_db():
    """
    Robust MongoDB connector:
    - Reads secrets from streamlit secrets (either nested dict or flat keys) or env vars.
    - Returns a connected database object or None (app shows helpful message instead of stopping).
    """
    uri = None
    dbname = None

    # 1) Try nested secret block: st.secrets["MONGO"]["URI"]
    try:
        if "MONGO" in st.secrets and isinstance(st.secrets["MONGO"], dict):
            uri = st.secrets["MONGO"].get("URI") or st.secrets["MONGO"].get("URI".lower())
            dbname = st.secrets["MONGO"].get("DBNAME") or st.secrets["MONGO"].get("DBNAME".lower()) or st.secrets["MONGO"].get("DB")
    except Exception:
        pass

    # 2) Try flat secrets keys: st.secrets["MONGO_URI"], st.secrets["MONGO_DBNAME"]
    try:
        if not uri:
            uri = st.secrets.get("MONGO_URI") or st.secrets.get("MONGO_URI".lower())
        if not dbname:
            dbname = st.secrets.get("MONGO_DBNAME") or st.secrets.get("MONGO_DB") or st.secrets.get("MONGO_DBNAME".lower())
    except Exception:
        pass

    # 3) Fallback to environment variables
    if not uri:
        uri = os.getenv("MONGO_URI") or os.getenv("MONGO")
    if not dbname:
        dbname = os.getenv("MONGO_DBNAME") or os.getenv("MONGO_DB") or os.getenv("MONGO_DBNAME")

    if not uri or not dbname:
        # return None — caller must handle missing DB gracefully
        st.warning("MongoDB not configured. Some features (persistent logging, inventory) will be disabled until you set MONGO_URI & MONGO_DBNAME in Streamlit Secrets or environment variables.")
        return None

    try:
        client = MongoClient(uri, tls=True, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=15000)
        client.admin.command("ping")
        return client[dbname]
    except Exception as e:
        st.warning("Failed to connect to MongoDB. App will continue in degraded mode (local CSV fallback).")
        # show debug to console only
        import traceback; traceback.print_exc()
        return None

# CALL
db = get_db()

# If db is None, set collection variables to None; rest of the app should handle None gracefully
if db:
    inv_col = db["inventory"]
    cons_col = db["consumables"]
    log_col = db["usage_log"]
    med_col = db["medicines"]
    consumables_col = cons_col
else:
    inv_col = cons_col = log_col = med_col = consumables_col = None

# --------------------------
# Authentication
# --------------------------
init_user_db()

if "authenticated" not in st.session_state:
    st.session_state.update({
        "authenticated": False,
        "username": None,
        "role": None,
        "last_active": None,
    })

def session_timeout():
    last = st.session_state["last_active"]
    if not last:
        return False
    return (datetime.now() - datetime.fromisoformat(last)).total_seconds() > SESSION_TIMEOUT_SECONDS

if not st.session_state["authenticated"]:
    login_router()
    st.stop()

if session_timeout():
    st.session_state["authenticated"] = False
    st.warning("Session timed out.")
    st.rerun()

st.session_state["last_active"] = datetime.now().isoformat()

# --------------------------
# UI Setup
# --------------------------
st.set_page_config(page_title="MedSafe AI", page_icon="💊", layout="wide")
apply_theme()
apply_layout_styles()
apply_global_css()
set_background("bg1.jpg")
show_logo("logo.png")

username = st.session_state["username"]
role = st.session_state["role"]

st.title("💊 Medicine Safety Comparator")

# --------------------------
# Load ML Data From MongoDB (SAFE MODE)
# --------------------------
docs = list(med_col.find({}))
if not docs:
    st.error("No ML records found in MongoDB 'medicines'.")
    st.stop()

df = pd.DataFrame(docs)

# Drop Mongo _id
df = df.drop(columns=["_id"], errors="ignore")

# Required columns for ML
required_cols = [
    "UPC", "Active Ingredient", "Disease/Use Case",
    "Days Until Expiry", "Storage Temperature (C)",
    "Dissolution Rate (%)", "Disintegration Time (minutes)",
    "Impurity Level (%)", "Assay Purity (%)",
    "Warning Labels Present", "Safe/Not Safe"
]

# Create missing columns
for c in required_cols:
    if c not in df.columns:
        df[c] = None

# Drop rows missing essential text fields
df = df.dropna(subset=["Active Ingredient", "Disease/Use Case", "Safe/Not Safe"], how="any")

# Reset index
df = df.reset_index(drop=True)

# Text cleanup
df["UPC"] = df["UPC"].astype(str).str.strip()
df["Active Ingredient"] = df["Active Ingredient"].fillna("Unknown").astype(str)
df["Disease/Use Case"] = df["Disease/Use Case"].fillna("Unknown").astype(str)
df["Safe/Not Safe"] = df["Safe/Not Safe"].fillna("Safe").astype(str)

# Numeric fields
num_cols = [
    "Days Until Expiry",
    "Storage Temperature (C)",
    "Dissolution Rate (%)",
    "Disintegration Time (minutes)",
    "Impurity Level (%)",
    "Assay Purity (%)",
    "Warning Labels Present"
]

# Convert ALL numeric fields, replace bad values with 0
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# Final X/Y
X = df[["Active Ingredient", "Disease/Use Case"] + num_cols]
y = df["Safe/Not Safe"]

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)

# Ensure at least 2 classes
if len(np.unique(y)) < 2:
    dummy = df.iloc[0].copy()
    dummy["Active Ingredient"] = "DummyUnsafe"
    dummy["Safe/Not Safe"] = "Not Safe"
    for col in num_cols:
        dummy[col] = 0
    df = pd.concat([df, pd.DataFrame([dummy])], ignore_index=True)
    y = le.fit_transform(df["Safe/Not Safe"])
    X = df[["Active Ingredient", "Disease/Use Case"] + num_cols]



# --------------------------
# Train ML Model
# --------------------------
def train_model(X, y):
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    pre = ColumnTransformer([
        ("ing", TfidfVectorizer(max_features=50), "Active Ingredient"),
        ("dis", TfidfVectorizer(max_features=50), "Disease/Use Case"),
        ("num", numeric_transformer, num_cols),
    ])

    pipe = Pipeline([
        ("preprocess", pre),
        ("model", LogisticRegression(max_iter=1000)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)
    return pipe

model = train_model(X, y)

# --------------------------
# Safety Rules
# --------------------------
SAFETY = {
    "Days Until Expiry": {"min": 30},
    "Storage Temperature (C)": {"range": (15, 30)},
    "Dissolution Rate (%)": {"min": 80},
    "Disintegration Time (minutes)": {"max": 30},
    "Impurity Level (%)": {"max": 2},
    "Assay Purity (%)": {"min": 90},
    "Warning Labels Present": {"min": 1},
}

def suggestions(vals):
    out = []
    for col, val in vals.items():
        rule = SAFETY.get(col)
        if not rule:
            continue
        if "min" in rule and val < rule["min"]:
            out.append(f"Increase *{col}* (min {rule['min']}).")
        if "max" in rule and val > rule["max"]:
            out.append(f"Reduce *{col}* (max {rule['max']}).")
        if "range" in rule:
            lo, hi = rule["range"]
            if not (lo <= val <= hi):
                out.append(f"Keep *{col}* within {lo}-{hi}.")
    return out

# --------------------------
# Sidebar Navigation
# --------------------------
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

# -------------------- SIDEBAR ROLE MENU --------------------
with st.sidebar:
    st.markdown("<h3 style='color:#2E86C1;margin-bottom:6px;'>MedSafe AI</h3>", unsafe_allow_html=True)
    render_avatar(st.session_state.get("username", "User"), size=72)
    st.sidebar.write(f"**{st.session_state.get('username','User')}**")
    st.sidebar.write(f"Role: **{st.session_state.get('role','guest')}**")
    st.sidebar.markdown("---")

    # Logout button
    if st.sidebar.button("Logout 🔒"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.session_state["role"] = None
        st.session_state["last_active"] = None
        st.success("Logged out. Redirecting to login...")
        st.rerun()

    role_normalized = (role or "").strip().lower()

    if role_normalized == "admin":
        allowed_tabs = ["📊 Dashboard", "📦 Inventory", "🔑 Change Password"]

    elif role_normalized == "pharmacist":
        allowed_tabs = ["📦 Inventory", "🧪 Testing", "🔑 Change Password"]

    else:
        allowed_tabs = ["📦 Inventory"]

    menu = st.sidebar.radio("📌 Navigation", allowed_tabs)

    st.sidebar.markdown("---")
    st.sidebar.write("© 2025 MedSafe AI")


# =========================================================
# 🧪 TESTING PAGE (robust + Mongo logging + PDF)
# =========================================================
if menu == "🧪 Testing":
    st.header("🧪 Medicine Safety Testing")

    # Use a form so page doesn't refresh mid-entry
    with st.form("testing_form"):
        col1, col2 = st.columns(2)
        upc_input = col1.text_input("Enter UPC").strip()
        ingr_input = col2.text_input("Enter Active Ingredient").strip()

        st.subheader("🏭 Competitor Details")
        comp_name = st.text_input("Competitor Name").strip()
        comp_gst = st.text_input("GST Number").strip()
        comp_addr = st.text_area("Address").strip()
        comp_phone = st.text_input("Phone").strip()

        # numeric inputs
        competitor_values = {}
        for c in num_cols:
            competitor_values[c] = st.number_input(c, value=0.0, format="%.6f")

        submitted = st.form_submit_button("🔎 Compare")

    selected = None
    result = None
    suggestions_list = []

    if submitted:
        # normalize search inputs
        upc_norm = (upc_input or "").strip()
        ingr_norm = (ingr_input or "").strip().lower()

        # prefer UPC lookup
        if upc_norm:
            # safe compare (make sure UPC column exists)
            if "UPC" in df.columns:
                match = df[df["UPC"].astype(str).str.strip() == upc_norm]
            else:
                match = df[df["UPC"].astype(str).str.lower().str.contains(upc_norm.lower(), na=False)]
            if not match.empty:
                selected = match.iloc[0].to_dict()
                ingr_input = selected.get("Active Ingredient", ingr_input)
                st.success(f"Found → Ingredient: {ingr_input}")
            else:
                st.error("UPC not found. Try the full or partial UPC or use Active Ingredient search.")
        elif ingr_norm:
            if "Active Ingredient" in df.columns:
                match = df[df["Active Ingredient"].fillna("").str.lower().str.strip() == ingr_norm]
            else:
                match = df[df["Active Ingredient"].fillna("").str.lower().str.contains(ingr_norm, na=False)]
            if not match.empty:
                selected = match.iloc[0].to_dict()
                upc_input = str(selected.get("UPC", ""))
                st.success(f"Found → UPC: {upc_input}")
            else:
                st.error("Ingredient not found.")
        else:
            st.error("Enter a UPC or an Active Ingredient to search first.")

        # if we found a standard record, build competitor DataFrame and predict
        if selected is not None:
            # Build competitor record ensuring all numeric cols present and are floats
            comp_record = {}
            for c in num_cols:
                try:
                    comp_record[c] = float(competitor_values.get(c, 0.0))
                except Exception:
                    comp_record[c] = 0.0

            comp_record["Active Ingredient"] = ingr_input or selected.get("Active Ingredient", "")
            # include Disease/Use Case if available in df
            comp_record["Disease/Use Case"] = selected.get("Disease/Use Case", "Unknown") if "Disease/Use Case" in df.columns else "Unknown"

            comp_df = pd.DataFrame([comp_record])

            # Ensure columns ordering matches the model features when possible
            try:
                if hasattr(model, "feature_names_in_"):
                    feat_names = list(model.feature_names_in_)
                    # Keep only features present in comp_df
                    feat_names = [f for f in feat_names if f in comp_df.columns]
                    comp_df = comp_df[feat_names]
            except Exception:
                pass

            # Predict with robust error handling
            try:
                pred = model.predict(comp_df)
                pred_val = pred[0] if hasattr(pred, "__len__") else pred
                if 'le' in globals() and hasattr(le, "inverse_transform"):
                    result = le.inverse_transform([pred_val])[0]
                else:
                    result = str(pred_val)
            except Exception:
                st.error("Prediction failed — the model could not process the input. Check features/dtypes in your dataset.")
                import traceback; traceback.print_exc()
                result = "ERROR"

            # Display prediction
            if result and result != "ERROR":
                if result.lower() == "safe":
                    st.success(f"Prediction → **{result}**")
                else:
                    st.error(f"Prediction → **{result}**")

            # Chart: Standard vs Competitor
            try:
                fig, ax = plt.subplots(figsize=(10, 4))
                x = np.arange(len(num_cols))
                std_vals = [float(selected.get(c, 0.0)) if selected.get(c, None) is not None else 0.0 for c in num_cols]
                comp_vals_plot = [float(comp_record.get(c, 0.0)) for c in num_cols]

                width = 0.35
                bars1 = ax.bar(x - width/2, std_vals, width=width, label="Standard")
                bars2 = ax.bar(x + width/2, comp_vals_plot, width=width, label="Competitor")
                ax.set_xticks(x)
                ax.set_xticklabels(num_cols, rotation=35, ha="right")
                ax.set_ylabel("Value")
                ax.set_title(f"Standard vs Competitor — {comp_name or 'Competitor'}")
                ax.legend()
                for bar in bars1 + bars2:
                    h = bar.get_height()
                    ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
                st.pyplot(fig)
            except Exception:
                st.warning("Could not draw comparison chart.")

            # Suggestions based on safety rules (if not safe)
            if isinstance(result, str) and result.lower() == "not safe":
                st.error("❌ Not Safe")
                try:
                    suggestions_list = suggestions(comp_record)
                    if suggestions_list:
                        st.subheader("Improvements")
                        for s in suggestions_list:
                            st.write("- ", s)
                except Exception:
                    st.warning("Could not compute suggestions.")

            # ---- LOG to MongoDB (with CSV fallback) ----
            log_doc = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "UPC": upc_input,
                "Ingredient": ingr_input,
                "Competitor": comp_name,
                "Competitor_GST": comp_gst,
                "Result": result
            }

            # try Mongo insert if log_col exists
            logged = False
            if log_col is not None:
                try:
                    log_col.insert_one(log_doc)
                    st.info("Result logged to MongoDB.")
                    logged = True
                except Exception:
                    st.warning("MongoDB write failed — will fallback to local CSV.")
                    import traceback; traceback.print_exc()

            # fallback CSV
            if not logged:
                try:
                    fallback_file = "test_logs.csv"
                    tmp_df = pd.DataFrame([log_doc])
                    if not os.path.exists(fallback_file):
                        tmp_df.to_csv(fallback_file, index=False)
                    else:
                        tmp_df.to_csv(fallback_file, mode="a", header=False, index=False)
                    st.info(f"Result logged to local CSV: {fallback_file}")
                except Exception:
                    st.error("Failed to log result to MongoDB or local CSV.")

            # --- PDF report (same pattern you had) ---
            try:
                import io
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
                from reportlab.lib.styles import getSampleStyleSheet
                from reportlab.lib.pagesizes import A4
                from reportlab.lib.utils import ImageReader

                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=A4)
                styles = getSampleStyleSheet()
                elements = []

                if os.path.exists("logo.png"):
                    try:
                        elements.append(RLImage("logo.png", width=100, height=100))
                        elements.append(Spacer(1, 12))
                    except Exception:
                        pass

                elements.append(Paragraph("💊 Medicine Safety Comparison Report", styles["Title"]))
                elements.append(Spacer(1, 12))
                elements.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
                elements.append(Spacer(1, 12))

                elements.append(Paragraph("<b>Standard Medicine</b>", styles["Heading2"]))
                elements.append(Paragraph(f"UPC: {upc_input}", styles["Normal"]))
                elements.append(Paragraph(f"Ingredient: {ingr_input}", styles["Normal"]))
                elements.append(Spacer(1, 12))

                elements.append(Paragraph("<b>Competitor Medicine</b>", styles["Heading2"]))
                elements.append(Paragraph(f"Name: {comp_name}", styles["Normal"]))
                elements.append(Paragraph(f"GST Number: {comp_gst}", styles["Normal"]))
                elements.append(Paragraph(f"Address: {comp_addr}", styles["Normal"]))
                elements.append(Paragraph(f"Phone: {comp_phone}", styles["Normal"]))
                elements.append(Spacer(1, 12))

                elements.append(Paragraph("<b>Prediction Result</b>", styles["Heading2"]))
                if result and result.lower() == "safe":
                    elements.append(Paragraph(f"<font color='green'><b>{result}</b></font>", styles["Normal"]))
                else:
                    elements.append(Paragraph(f"<font color='red'><b>{result}</b></font>", styles["Normal"]))
                elements.append(Spacer(1, 12))

                if result and result.lower() == "not safe" and suggestions_list:
                    elements.append(Paragraph("<b>⚠️ Suggested Improvements:</b>", styles["Heading2"]))
                    for s in suggestions_list:
                        elements.append(Paragraph(f"- {s}", styles["Normal"]))
                    elements.append(Spacer(1, 12))

                # attach chart if available
                try:
                    chart_buffer = io.BytesIO()
                    fig.savefig(chart_buffer, format="png", bbox_inches="tight")
                    chart_buffer.seek(0)
                    img_reader = ImageReader(chart_buffer)
                    elements.append(RLImage(img_reader, width=400, height=250))
                    elements.append(Spacer(1, 12))
                except Exception:
                    pass

                doc.build(elements)
                buffer.seek(0)

                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=buffer.getvalue(),
                    file_name=f"Medicine_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
            except Exception:
                st.warning("Failed to generate PDF report.")


# =========================================================
# 📊 DASHBOARD
# =========================================================
elif menu == "📊 Dashboard":
    st.header("📊 Dashboard")
    logs = list(log_col.find({}).sort([("_id", -1)]))
    if not logs:
        st.info("No logs yet.")
    else:
        logs_df = pd.DataFrame(logs)
        logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"])

        total = len(logs_df)
        safe = logs_df["Result"].str.lower().eq("safe").sum()
        unsafe = logs_df["Result"].str.lower().eq("not safe").sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Tests", total)
        col2.metric("Safe", safe)
        col3.metric("Unsafe", unsafe)

        trend = logs_df.groupby(logs_df["timestamp"].dt.date).size()
        st.line_chart(trend)

        st.subheader("Recent 10 Tests")
        st.dataframe(logs_df.head(10)[["timestamp", "UPC", "Ingredient", "Competitor", "Result"]])



# =========================================================
# 📦 INVENTORY PAGE (Medicines + Consumables Tabs) with Filters
# =========================================================
elif menu == "📦 Inventory":

    st.header("📦 Inventory Management")

    # -------------------------------------------
    # LOAD MEDICINES
    # -------------------------------------------
    def load_medicines():
        docs = list(inv_col.find({}))
        for d in docs:
            d["_id"] = str(d["_id"])
            # ensure expected keys exist
            for k in ["UPC", "Ingredient", "Manufacturer", "Batch", "Stock", "Expiry"]:
                if k not in d:
                    d[k] = None
        return pd.DataFrame(docs)

    # -------------------------------------------
    # LOAD CONSUMABLES
    # -------------------------------------------
    def load_consumables():
        docs = list(consumables_col.find({}))
        for d in docs:
            d["_id"] = str(d["_id"])
            for k in ["Item Name", "Category", "UPC", "Quantity in Stock",
                      "Expiry Period (Months)", "Storage Temperature (C)", "Safe/Not Safe"]:
                if k not in d:
                    d[k] = None
        return pd.DataFrame(docs)

    meds = load_medicines()
    cons = load_consumables()

    tab1, tab2 = st.tabs(["💊 Medicines", "🛠 Consumables"])

    # ------------------------------
    # Helper: apply filters to a DF
    # ------------------------------
    def apply_medicine_filters(df):
        # UI controls
        with st.expander("🔎 Filters", expanded=False):
            col1, col2, col3 = st.columns(3)
            f_upc = col1.text_input("Filter UPC", value="")
            f_ing = col1.text_input("Filter Ingredient", value="")
            f_batch = col2.text_input("Filter Batch", value="")
            f_mf = col2.text_input("Filter Manufacturer", value="")
            f_low_stock = col3.number_input("Low stock threshold (≤)", min_value=0, value=0, step=1)
            # apply filters
            df_filtered = df.copy()
            if f_upc.strip():
                df_filtered = df_filtered[df_filtered["UPC"].astype(str).str.contains(f_upc.strip(), case=False, na=False)]
            if f_ing.strip():
                df_filtered = df_filtered[df_filtered["Ingredient"].astype(str).str.contains(f_ing.strip(), case=False, na=False)]
            if f_batch.strip():
                df_filtered = df_filtered[df_filtered["Batch"].astype(str).str.contains(f_batch.strip(), case=False, na=False)]
            if f_mf.strip():
                df_filtered = df_filtered[df_filtered["Manufacturer"].astype(str).str.contains(f_mf.strip(), case=False, na=False)]
            if f_low_stock > 0:
                if "Stock" in df_filtered.columns:
                    df_filtered["Stock"] = pd.to_numeric(df_filtered["Stock"], errors="coerce").fillna(0).astype(int)
                    df_filtered = df_filtered[df_filtered["Stock"] <= int(f_low_stock)]
            return df_filtered

    def apply_consumable_filters(df):
        with st.expander("🔎 Filters", expanded=False):
            col1, col2, col3 = st.columns(3)
            f_name = col1.text_input("Filter Item Name", value="")
            f_cat = col1.text_input("Filter Category", value="")
            f_upc = col2.text_input("Filter UPC", value="")
            f_safe = col3.selectbox("Safe / Not Safe", ["All", "Safe", "Not Safe"])
            f_low_qty = col3.number_input("Low qty threshold (≤)", min_value=0, value=0, step=1)
            df_filtered = df.copy()
            if f_name.strip():
                df_filtered = df_filtered[df_filtered["Item Name"].astype(str).str.contains(f_name.strip(), case=False, na=False)]
            if f_cat.strip():
                df_filtered = df_filtered[df_filtered["Category"].astype(str).str.contains(f_cat.strip(), case=False, na=False)]
            if f_upc.strip():
                df_filtered = df_filtered[df_filtered["UPC"].astype(str).str.contains(f_upc.strip(), case=False, na=False)]
            if f_safe != "All":
                df_filtered = df_filtered[df_filtered["Safe/Not Safe"] == f_safe]
            if f_low_qty > 0:
                if "Quantity in Stock" in df_filtered.columns:
                    df_filtered["Quantity in Stock"] = pd.to_numeric(df_filtered["Quantity in Stock"], errors="coerce").fillna(0).astype(int)
                    df_filtered = df_filtered[df_filtered["Quantity in Stock"] <= int(f_low_qty)]
            return df_filtered

    # ===========================
    # TAB 1 → MEDICINES
    # ===========================
    with tab1:
        st.subheader("💊 Medicine Inventory")

        total_items = len(meds)
        total_stock = meds["Stock"].fillna(0).astype(int).sum() if not meds.empty else 0
        c1, c2 = st.columns(2)
        c1.metric("Total Medicines", total_items)
        c2.metric("Total Stock", int(total_stock))

        # Add / Update form (same as before)
        st.markdown("### ➕ Add / Update Medicine")
        with st.form("add_med"):
            colA, colB, colC = st.columns(3)
            upc = colA.text_input("UPC")
            ing = colA.text_input("Ingredient")
            mf = colB.text_input("Manufacturer")
            batch = colB.text_input("Batch")
            stock = colC.number_input("Stock", min_value=0, value=1)
            expiry = colC.date_input("Expiry Date")
            if st.form_submit_button("Save Medicine"):
                if not upc.strip():
                    st.error("UPC is required.")
                else:
                    doc = {
                        "UPC": upc.strip(),
                        "Ingredient": ing.strip(),
                        "Manufacturer": mf.strip(),
                        "Batch": batch.strip(),
                        "Stock": int(stock),
                        "Expiry": expiry.isoformat()
                    }
                    existing = inv_col.find_one({"UPC": doc["UPC"], "Batch": doc["Batch"]})
                    if existing:
                        inv_col.update_one({"_id": existing["_id"]}, {"$set": doc})
                    else:
                        inv_col.insert_one(doc)
                    st.success("Saved successfully.")
                    st.rerun()

        st.markdown("### 📋 Medicine List (use Filters to narrow results)")
        meds_filtered = apply_medicine_filters(meds) if not meds.empty else meds

        if meds_filtered.empty:
            st.info("No medicines found for the selected filters.")
        else:
            show = meds_filtered.copy().drop(columns=["_id"], errors="ignore")
            if "Expiry" in show.columns:
                show["Expiry"] = pd.to_datetime(show["Expiry"], errors="coerce").dt.date
            st.dataframe(show, use_container_width=True)

            # selection uses filtered list (map to original _id)
            labels = [f"{r.get('Ingredient','(no name)')} | UPC:{r.get('UPC','')} | Batch:{r.get('Batch','')}" for _, r in meds_filtered.iterrows()]
            idx_list = list(meds_filtered.index)
            sel_idx = st.selectbox("Select medicine", options=list(range(len(labels))), format_func=lambda i: labels[i])
            real_idx = idx_list[sel_idx]
            rec = meds_filtered.loc[real_idx]
            sel_id = rec["_id"]

            col1, col2, col3 = st.columns(3)
            new_ing = col1.text_input("Ingredient", rec.get("Ingredient", ""))
            new_upc = col1.text_input("UPC", rec.get("UPC", ""))
            new_mf = col2.text_input("Manufacturer", rec.get("Manufacturer", ""))
            new_batch = col2.text_input("Batch", rec.get("Batch", ""))
            new_stock = col3.number_input("Stock", min_value=0, value=int(rec.get("Stock") or 0))
            try:
                old_exp = pd.to_datetime(rec.get("Expiry")).date()
            except:
                old_exp = datetime.today().date()
            new_exp = col3.date_input("Expiry", old_exp)

            if st.button("Save Medicine Changes"):
                inv_col.update_one({"_id": ObjectId(sel_id)}, {"$set": {
                    "Ingredient": new_ing.strip(),
                    "UPC": new_upc.strip(),
                    "Manufacturer": new_mf.strip(),
                    "Batch": new_batch.strip(),
                    "Stock": int(new_stock),
                    "Expiry": new_exp.isoformat()
                }})
                st.success("Updated.")
                st.rerun()

            if st.button("Delete Medicine"):
                inv_col.delete_one({"_id": ObjectId(sel_id)})
                st.success("Deleted.")
                st.rerun()

    # ===========================
    # TAB 2 → CONSUMABLES
    # ===========================
    with tab2:
        st.subheader("🛠 Consumables Inventory")

        total_items = len(cons)
        total_qty = cons["Quantity in Stock"].fillna(0).astype(int).sum() if not cons.empty else 0
        c1, c2 = st.columns(2)
        c1.metric("Total Consumables", total_items)
        c2.metric("Total Quantity", int(total_qty))

        st.markdown("### ➕ Add / Update Consumable")
        with st.form("add_cons"):
            colA, colB = st.columns(2)
            name = colA.text_input("Item Name")
            category = colA.text_input("Category")
            upc = colB.text_input("UPC")
            qty = colB.number_input("Quantity", min_value=0, value=1)
            expiry_m = colA.number_input("Expiry (Months)", min_value=0, value=12)
            storage = colB.number_input("Storage Temp (°C)", value=25)
            safe_flag = colA.selectbox("Safe / Not Safe", ["Safe", "Not Safe"])
            if st.form_submit_button("Save Consumable"):
                doc = {
                    "Item Name": name.strip(),
                    "Category": category.strip(),
                    "UPC": upc.strip(),
                    "Quantity in Stock": int(qty),
                    "Expiry Period (Months)": int(expiry_m),
                    "Storage Temperature (C)": storage,
                    "Safe/Not Safe": safe_flag
                }
                existing = consumables_col.find_one({"UPC": doc["UPC"]}) if doc["UPC"] else None
                if existing:
                    consumables_col.update_one({"_id": existing["_id"]}, {"$set": doc})
                else:
                    consumables_col.insert_one(doc)
                st.success("Consumable saved.")
                st.rerun()

        st.markdown("### 📋 Consumables List (use Filters to narrow results)")
        cons_filtered = apply_consumable_filters(cons) if not cons.empty else cons

        if cons_filtered.empty:
            st.info("No consumables found for the selected filters.")
        else:
            show2 = cons_filtered.copy().drop(columns=["_id"], errors="ignore")
            st.dataframe(show2, use_container_width=True)

            labels = [f"{r.get('Item Name','(no name)')} | UPC:{r.get('UPC','')} | Qty:{int(r.get('Quantity in Stock') or 0)}" for _, r in cons_filtered.iterrows()]
            idx_list = list(cons_filtered.index)
            sel_idx = st.selectbox("Select consumable", options=list(range(len(labels))), format_func=lambda i: labels[i])
            real_idx = idx_list[sel_idx]
            rec = cons_filtered.loc[real_idx]
            sel_id = rec["_id"]

            col1, col2 = st.columns(2)
            new_item = col1.text_input("Item Name", rec.get("Item Name", ""))
            new_cat = col1.text_input("Category", rec.get("Category", ""))
            new_upc = col2.text_input("UPC", rec.get("UPC", ""))
            new_qty = col2.number_input("Quantity", min_value=0, value=int(rec.get("Quantity in Stock") or 0))
            new_exp = col1.number_input("Expiry (Months)", min_value=0, value=int(rec.get("Expiry Period (Months)") or 0))
            new_safe = col2.selectbox("Safe / Not Safe", ["Safe", "Not Safe"], index=0 if rec.get("Safe/Not Safe","Safe")=="Safe" else 1)

            if st.button("Save Consumable Changes"):
                consumables_col.update_one({"_id": ObjectId(sel_id)}, {"$set": {
                    "Item Name": new_item.strip(),
                    "Category": new_cat.strip(),
                    "UPC": new_upc.strip(),
                    "Quantity in Stock": int(new_qty),
                    "Expiry Period (Months)": int(new_exp),
                    "Safe/Not Safe": new_safe
                }})
                st.success("Updated.")
                st.rerun()

            if st.button("Delete Consumable"):
                consumables_col.delete_one({"_id": ObjectId(sel_id)})
                st.success("Deleted.")
                st.rerun()


# =========================================================
# 🔑 CHANGE PASSWORD PAGE
# =========================================================
elif menu == "🔑 Change Password":
    password_reset(username)

# =========================================================
# END OF FILE
# =========================================================
