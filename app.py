
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
# MongoDB Connection
# --------------------------
@st.cache_resource
def get_db():
    try:
        uri = st.secrets["MONGO"]["URI"]
        dbname = st.secrets["MONGO"]["DBNAME"]
    except:
        uri = os.getenv("MONGO_URI")
        dbname = os.getenv("MONGO_DBNAME")

    if not uri or not dbname:
        st.error("Missing MongoDB configuration.")
        st.stop()

    client = MongoClient(
        uri,
        tls=True,
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=15000,
    )

    client.admin.command("ping")
    return client[dbname]

db = get_db()

# Collections
inv_col = db["inventory"]
cons_col = db["consumables"]
log_col = db["usage_log"]
med_col = db["medicines"]

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

    role_normalized = role.strip().lower()

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
# 🧪 TESTING PAGE
# =========================================================
if menu == "🧪 Testing":
    st.header("🧪 Medicine Safety Testing")
    col1, col2 = st.columns(2)
    upc_input = col1.text_input("Enter UPC")
    ingr_input = col2.text_input("Enter Active Ingredient")

    selected = None

    # ------ Search logic ------
    if upc_input:
        match = df[df["UPC"] == upc_input]
        if not match.empty:
            selected = match.iloc[0]
            ingr_input = selected["Active Ingredient"]
            st.success(f"Found → Ingredient: {ingr_input}")
        else:
            st.error("UPC not found.")

    elif ingr_input:
        match = df[df["Active Ingredient"].str.lower() == ingr_input.lower()]
        if not match.empty:
            selected = match.iloc[0]
            upc_input = selected["UPC"]
            st.success(f"Found → UPC: {upc_input}")
        else:
            st.error("Ingredient not found.")

    st.subheader("🏭 Competitor Details")
    comp_name = st.text_input("Competitor Name")
    comp_gst = st.text_input("GST Number")
    comp_addr = st.text_area("Address")
    comp_phone = st.text_input("Phone")

    comp_vals = {c: st.number_input(c, value=0.0) for c in num_cols}

    if st.button("🔎 Compare"):
        if selected is None:
            st.error("Enter valid UPC or Ingredient first.")
        else:
            comp_df = pd.DataFrame([{
                "Active Ingredient": ingr_input,
                "Disease/Use Case": "Unknown",
                **comp_vals
            }])

            pred = model.predict(comp_df)[0]
            res = le.inverse_transform([pred])[0]

            st.success(f"Prediction → **{res}**")

            # Chart
            fig, ax = plt.subplots(figsize=(10, 4))
            x = np.arange(len(num_cols))
            ax.bar(x - 0.3, [selected[c] for c in num_cols], width=0.3, label="Standard")
            ax.bar(x + 0.3, [comp_vals[c] for c in num_cols], width=0.3, label="Competitor")
            ax.set_xticks(x)
            ax.set_xticklabels(num_cols, rotation=35, ha="right")
            ax.legend()
            st.pyplot(fig)

            # Suggestions
            if res.lower() == "not safe":
                st.error("❌ Not Safe")
                sug = suggestions(comp_vals)
                if sug:
                    st.subheader("Improvements")
                    for s in sug:
                        st.write("- ", s)

            # Log result
            log_col.insert_one({
                "timestamp": datetime.now().isoformat(),
                "UPC": upc_input,
                "Ingredient": ingr_input,
                "Competitor": comp_name,
                "Result": res
            })

# =========================================================
# 📊 DASHBOARD
# =========================================================
elif menu == "📊 Dashboard":
    st.header("📊 Dashboard")
    logs = list(log_col.find({}).sort("_id", -1))
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
