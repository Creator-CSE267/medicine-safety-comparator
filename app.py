
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
import joblib
import pathlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
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

# alias for older code that uses `consumables_col`
consumables_col = cons_col

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
set_background("bg4.jpg")
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
# Train / Load ML Model (Silent Mode)
# --------------------------

MODEL_DIR = pathlib.Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "std_delta_pipeline.joblib"
LABEL_PATH = MODEL_DIR / "std_label_encoder.joblib"

model = None

# 1. Try loading saved model first (silent)
try:
    if MODEL_PATH.exists() and LABEL_PATH.exists():
        pipeline = joblib.load(MODEL_PATH)
        le = joblib.load(LABEL_PATH)
        model = pipeline
except Exception:
    model = None

# 2. Compute standard stats
std_stats = df.groupby("Active Ingredient")[num_cols].median()
global_std = std_stats.median() if not std_stats.empty else df[num_cols].median()

# 3. Build augmented DF
def build_augmented_df(raw_df):
    aug = raw_df.copy().reset_index(drop=True)
    aug = aug.merge(std_stats, how="left",
                    left_on="Active Ingredient",
                    right_index=True,
                    suffixes=("", "_std"))

    global_meds = std_stats.median().to_dict() if not std_stats.empty else df[num_cols].median().to_dict()

    for c in num_cols:
        std_col = c + "_std"
        if std_col not in aug.columns:
            aug[std_col] = aug[c].map(global_meds).fillna(0)
        else:
            aug[std_col] = aug[std_col].fillna(global_meds.get(c, 0))
        aug["d_" + c] = aug[c].astype(float) - aug[std_col].astype(float)

    aug = aug.drop(columns=[c + "_std" for c in num_cols], errors="ignore")
    return aug

df_aug = build_augmented_df(df)

# Prepare features
text_cols = ["Active Ingredient", "Disease/Use Case"]
abs_cols = list(num_cols)
delta_cols = ["d_" + c for c in num_cols]
feature_cols = text_cols + abs_cols + delta_cols

X_aug = df_aug[feature_cols]
y_aug = df_aug["Safe/Not Safe"]

# Encode target
le = LabelEncoder()
y_enc = le.fit_transform(y_aug)

# Build pipeline (no UI messages)
numeric_feature_list = abs_cols + delta_cols
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

preproc = ColumnTransformer([
    ("tf_ing", TfidfVectorizer(max_features=100, ngram_range=(1,2)), "Active Ingredient"),
    ("tf_dis", TfidfVectorizer(max_features=100, ngram_range=(1,2)), "Disease/Use Case"),
    ("num", numeric_transformer, numeric_feature_list),
], remainder="drop")

pipeline = Pipeline([
    ("preprocess", preproc),
    ("clf", RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=42, n_jobs=-1
    ))
])

# 4. Train model only if not loaded
if model is None:
    try:
        strat = y_enc if len(np.unique(y_enc)) > 1 and len(y_enc) >= 4 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_aug, y_enc, test_size=0.2,
            random_state=42, stratify=strat
        )
        pipeline.fit(X_train, y_train)

        # Try calibration silently
        try:
            X_train_trans = pipeline.named_steps["preprocess"].transform(X_train)
            calib = CalibratedClassifierCV(
                pipeline.named_steps["clf"], method="isotonic", cv=3
            )
            calib.fit(X_train_trans, y_train)
            pipeline.named_steps["clf"] = calib
        except Exception:
            pass

        # Save silently
        try:
            joblib.dump(pipeline, MODEL_PATH)
            joblib.dump(le, LABEL_PATH)
        except Exception:
            pass

        model = pipeline

    except Exception:
        # Training failed silently → use fallback model
        from sklearn.dummy import DummyClassifier
        dummy = DummyClassifier(strategy="most_frequent")
        try:
            dummy.fit(X_aug.fillna(0), y_enc)
            model = dummy
        except Exception:
            model = None



# --------------------------
# Safety Rules (including Assay optimal zone)
# --------------------------
SAFETY = {
    "Days Until Expiry": {"min": 30},
    "Storage Temperature (C)": {"range": (15, 30)},
    "Dissolution Rate (%)": {"min": 80},
    "Disintegration Time (minutes)": {"max": 30},
    "Impurity Level (%)": {"max": 2},
    # Set both min and max for Assay Purity - replace 90/105 with pharmacopeial min/max if you know them
    "Assay Purity (%)": {"min": 90, "max": 105},
    "Warning Labels Present": {"min": 1},
}

# convenience sets to decide direction
HIGHER_BETTER = {"Days Until Expiry", "Dissolution Rate (%)", "Warning Labels Present"}
LOWER_BETTER = {"Disintegration Time (minutes)", "Impurity Level (%)"}
# Assay Purity is handled as an optimal zone

def suggestions(vals):
    """
    Generate human-friendly suggestions using SAFETY rules.
    vals is a dict of numeric values (competitor values).
    """
    out = []
    for col, val in vals.items():
        try:
            val_f = float(val)
        except Exception:
            continue
        rule = SAFETY.get(col)
        if not rule:
            continue
        # explicit min/max
        if "min" in rule and val_f < rule["min"]:
            out.append(f"Increase **{col}** to at least {rule['min']}. (Current: {val_f})")
        if "max" in rule and val_f > rule["max"]:
            out.append(f"Reduce **{col}** to at most {rule['max']}. (Current: {val_f})")
        if "range" in rule:
            lo, hi = rule["range"]
            if not (lo <= val_f <= hi):
                out.append(f"Keep **{col}** within {lo}–{hi}. (Current: {val_f})")

        # special case messaging for Assay Purity
        if col == "Assay Purity (%)":
            if "min" in rule and "max" in rule:
                if val_f < rule["min"]:
                    out.append(f"Assay Purity is low — review API content or manufacturing process to reach at least {rule['min']}%.")
                elif val_f > rule["max"]:
                    out.append(f"Assay Purity is above the acceptable upper limit — check for over-concentration or analytical error; target {rule['min']}–{rule['max']}%.")

    return out
def predict_rule_based(active_ingredient, competitor_values, std_row=None):
    """
    Compare competitor_values against standard (std_row) and SAFETY rules.
    Strict: result = "Safe" if fail_count == 0 else "Not Safe".
    Returns dict with details, pass/fail counts, and result.
    """
    # standard medians for ingredient
    if std_row is None:
        try:
            std_row = std_stats.loc[active_ingredient]
        except Exception:
            std_row = global_std

    pass_count = 0
    fail_count = 0
    details = []

    for col in num_cols:
        std_val = float(std_row.get(col, 0.0)) if hasattr(std_row, "get") else float(std_row[col])
        comp_val = float(competitor_values.get(col, 0.0))

        # Assay Purity optimal-zone handling
        if col == "Assay Purity (%)":
            rule = SAFETY.get(col, {})
            min_allowed = rule.get("min", None)
            max_allowed = rule.get("max", None)
            if min_allowed is not None and max_allowed is not None:
                if min_allowed <= comp_val <= max_allowed:
                    pass_count += 1
                    details.append(f"{col}: {comp_val} within acceptable range {min_allowed}-{max_allowed} → PASS")
                else:
                    fail_count += 1
                    details.append(f"{col}: {comp_val} outside acceptable range {min_allowed}-{max_allowed} → FAIL")
            else:
                # fallback: allow up to +5% above standard median (conservative)
                fallback_max = min(std_val * 1.05, 110.0)
                fallback_min = rule.get("min", std_val * 0.99)
                if fallback_min <= comp_val <= fallback_max:
                    pass_count += 1
                    details.append(f"{col}: {comp_val} within fallback range {fallback_min:.2f}-{fallback_max:.2f} → PASS")
                else:
                    fail_count += 1
                    details.append(f"{col}: {comp_val} outside fallback range {fallback_min:.2f}-{fallback_max:.2f} → FAIL")
            continue

        # normal rules
        if col in HIGHER_BETTER:
            if comp_val >= std_val:
                pass_count += 1
                details.append(f"{col}: {comp_val} ≥ {std_val} → PASS")
            else:
                fail_count += 1
                details.append(f"{col}: {comp_val} < {std_val} → FAIL")
            continue

        if col in LOWER_BETTER:
            if comp_val <= std_val:
                pass_count += 1
                details.append(f"{col}: {comp_val} ≤ {std_val} → PASS")
            else:
                fail_count += 1
                details.append(f"{col}: {comp_val} > {std_val} → FAIL")
            continue

        # default: higher is better
        if comp_val >= std_val:
            pass_count += 1
            details.append(f"{col}: {comp_val} ≥ {std_val} → PASS")
        else:
            fail_count += 1
            details.append(f"{col}: {comp_val} < {std_val} → FAIL")

    total = pass_count + fail_count if (pass_count + fail_count) > 0 else 1
    confidence = pass_count / total

    result = "Safe" if (fail_count == 0 and pass_count > 0) else "Not Safe"

    return {
        "result": result,
        "confidence": confidence,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "details": details,
        "std_values": (std_row.to_dict() if hasattr(std_row, "to_dict") else dict(std_row))
    }


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
# 🧪 TESTING PAGE (REPLACEMENT)
# =========================================================
if menu == "🧪 Testing":
    st.header("🧪 Medicine Safety Testing")
    st.subheader("🔍 Search by UPC or Active Ingredient")

    col1, col2 = st.columns(2)
    with col1:
        upc_input = st.text_input("Enter UPC")
    with col2:
        ingr_input = st.text_input("Enter Active Ingredient")

    selected = None

    # ------ Search logic ------
    if upc_input:
        match = df[df["UPC"].astype(str).str.strip() == str(upc_input).strip()]
        if not match.empty:
            selected = match.iloc[0]
            ingr_input = selected["Active Ingredient"]
            st.success(f"Found → Ingredient: {ingr_input}")
        else:
            st.error("UPC not found.")
    elif ingr_input:
        match = df[df["Active Ingredient"].astype(str).str.lower().str.strip() == ingr_input.lower().strip()]
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

    # -------------------------
    # Competitor numeric inputs — START EMPTY (user must fill)
    # -------------------------
    comp_vals = {}
    for c in num_cols:
        # default value is 0.0 so user must provide values (no auto-populate from standard)
        comp_vals[c] = st.number_input(f"{c}", value=0.0, format="%.2f")

    # -------------------------
    # -------------------------
# Helper: build competitor DataFrame for model
# (keep this defined inside the Testing page)
# -------------------------
def build_comp_df(active_ing, disease, numeric_dict):
    rec = {"Active Ingredient": active_ing or "Unknown", "Disease/Use Case": disease or "Unknown"}
    for cc in num_cols:
        try:
            rec[cc] = float(numeric_dict.get(cc, 0.0))
        except Exception:
            rec[cc] = 0.0
    return pd.DataFrame([rec])

# -------------------------
# Compare button: run rule-based comparator, show suggestions, chart, logging, PDF
# (This must be INSIDE the Testing page block)
# -------------------------
if st.button("🔎 Compare"):
    # validate input
    if selected is None and (not ingr_input or ingr_input.strip() == ""):
        st.error("⚠️ Please enter a valid UPC or Active Ingredient first.")
    else:
        # Use ingredient text (user provided or selected)
        active_ing = ingr_input if ingr_input else (selected.get("Active Ingredient") if selected is not None else "Unknown")
        disease = selected.get("Disease/Use Case", "Unknown") if selected is not None else "Unknown"

        # Build competitor dict from comp_vals (user inputs)
        competitor_dict = {}
        for c in num_cols:
            try:
                competitor_dict[c] = float(comp_vals.get(c, 0.0))
            except Exception:
                competitor_dict[c] = 0.0

        # Get std_row for this active ingredient (fallback to global)
        try:
            std_row = std_stats.loc[active_ing]
        except Exception:
            std_row = global_std

        # Run the strict rule-based comparator (your deterministic rules)
        rule_out = predict_rule_based(active_ing, competitor_dict, std_row=std_row)
        result = rule_out["result"]
        details = rule_out["details"]
        confidence = rule_out["confidence"]

        # Display result + details
        if result.lower() == "safe":
            st.success(f"✅ Prediction → {result} (confidence {confidence:.2f})")
        else:
            st.error(f"❌ Prediction → {result} (confidence {confidence:.2f})")

        st.markdown("### 🔎 Per-criterion details")
        for d in details:
            # color-coded lines
            if "PASS" in d:
                st.markdown(f"<div style='color:green'>{d}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='color:red'>{d}</div>", unsafe_allow_html=True)

        # Generate suggestions from rule-based check (based on competitor values)
        suggestions_list = suggestions(competitor_dict)  # ensure this is defined BEFORE using
        if suggestions_list:
            # show suggestions in black color (user request)
            st.subheader("🔧 Suggested Improvements")
            for s in suggestions_list:
                st.markdown(f"<div style='color:#111111'>{s}</div>", unsafe_allow_html=True)

        # Chart compare standard vs competitor
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(num_cols))
        std_vals = [float(std_row.get(c, 0.0) or 0.0) for c in num_cols]
        comp_vals_list = [competitor_dict[c] for c in num_cols]

        ax.bar(x - 0.3, std_vals, width=0.3, label="Standard")
        ax.bar(x + 0.3, comp_vals_list, width=0.3, label="Competitor")
        ax.set_xticks(x)
        ax.set_xticklabels(num_cols, rotation=35, ha="right")
        ax.legend()
        st.pyplot(fig)

        # Log result (try Mongo first, else CSV)
        log_doc = {
            "timestamp": datetime.now().isoformat(),
            "UPC": upc_input,
            "Ingredient": active_ing,
            "Competitor": comp_name,
            "Result": result,
            "confidence": confidence,
            "details": details,
            "suggestions": suggestions_list
        }
        logged = False
        try:
            log_col.insert_one(log_doc)
            logged = True
            st.info("Logged result to MongoDB.")
        except Exception:
            st.warning("MongoDB logging failed — falling back to CSV.")
        if not logged:
            try:
                lf = "test_logs.csv"
                pd.DataFrame([log_doc]).to_csv(lf, mode="a", header=not os.path.exists(lf), index=False)
                st.info(f"Logged result to CSV: {lf}")
            except Exception:
                st.error("Failed to log result.")

        # Build PDF (include suggestions_list)
        try:
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.utils import ImageReader
            import io

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
            elements.append(Paragraph(f"UPC: {selected['UPC'] if selected is not None else 'N/A'}", styles["Normal"]))
            elements.append(Paragraph(f"Ingredient: {active_ing}", styles["Normal"]))
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("<b>Competitor Medicine</b>", styles["Heading2"]))
            elements.append(Paragraph(f"Name: {comp_name}", styles["Normal"]))
            elements.append(Paragraph(f"GST Number: {comp_gst}", styles["Normal"]))
            elements.append(Paragraph(f"Address: {comp_addr}", styles["Normal"]))
            elements.append(Paragraph(f"Phone: {comp_phone}", styles["Normal"]))
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("<b>Prediction Result</b>", styles["Heading2"]))
            if result.lower() == "safe":
                elements.append(Paragraph(f"<font color='green'><b>{result}</b></font>", styles["Normal"]))
            else:
                elements.append(Paragraph(f"<font color='red'><b>{result}</b></font>", styles["Normal"]))
            elements.append(Spacer(1, 12))

            if suggestions_list:
                elements.append(Paragraph("<b>⚠️ Suggested Improvements:</b>", styles["Heading2"]))
                for s in suggestions_list:
                    elements.append(Paragraph(f"- {s}", styles["Normal"]))
                elements.append(Spacer(1, 12))

            elements.append(Paragraph("<b>Criteria Comparison (Standard vs Competitor)</b>", styles["Heading2"]))
            for idx, c in enumerate(num_cols):
                elements.append(Paragraph(f"{c}: Standard = {std_vals[idx]}  |  Competitor = {comp_vals_list[idx]}", styles["Normal"]))
            elements.append(Spacer(1, 12))

            # attach chart
            chart_buffer = io.BytesIO()
            fig.savefig(chart_buffer, format="png", bbox_inches="tight")
            chart_buffer.seek(0)
            try:
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
        except Exception as e:
            st.warning("Failed to generate PDF report: " + str(e))




# =========================================================
# 📊 DASHBOARD (professional, robust, interactive)
# =========================================================
elif menu == "📊 Dashboard":
    st.header("📊 Test Results Dashboard")
    st.markdown(
        "Overview of recent medicine safety comparisons. "
        "Use filters to narrow results, inspect details, or download logs."
    )

    # --- Load logs safely ---
    try:
        raw_logs = list(log_col.find({}).sort([("_id", -1)]))
    except Exception as e:
        raw_logs = []
        st.warning("Unable to load logs from MongoDB: " + str(e))

    if not raw_logs:
        st.info("No logs found. Run some tests from the Testing page to populate logs.")
    else:
        # Convert to DataFrame and normalize fields
        logs_df = pd.DataFrame(raw_logs)

        # Ensure timestamp exists and parse robustly
        if "timestamp" in logs_df.columns:
            try:
                logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"], errors="coerce")
            except Exception:
                logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"].astype(str), errors="coerce")
        else:
            logs_df["timestamp"] = pd.NaT

        # Normalize common columns (fallback names)
        logs_df["UPC"] = logs_df.get("UPC", logs_df.get("upc", ""))
        logs_df["Ingredient"] = logs_df.get("Ingredient", logs_df.get("ingredient", ""))
        logs_df["Competitor"] = logs_df.get("Competitor", logs_df.get("competitor", ""))
        logs_df["Result"] = logs_df.get("Result", logs_df.get("result", "Unknown"))

        # Normalize suggestions if present (list -> string)
        def _format_sugg(v):
            if v is None:
                return ""
            if isinstance(v, (list, tuple)):
                return " | ".join(map(str, v))
            # sometimes stored as dict or str
            if isinstance(v, dict):
                # join dict values
                return " | ".join(f"{k}: {v[k]}" for k in v)
            return str(v)

        logs_df["suggestions_text"] = _format_sugg  # placeholder to be used by apply
        logs_df["suggestions_text"] = logs_df.get("suggestions", logs_df.get("suggestions_text", "")) \
            .apply(_format_sugg) if "suggestions" in logs_df.columns else ""

        # Provide nice KPIs
        total_tests = len(logs_df)
        safe_count = logs_df["Result"].astype(str).str.lower().eq("safe").sum()
        unsafe_count = logs_df["Result"].astype(str).str.lower().eq("not safe").sum()

        k1, k2, k3 = st.columns([1.2, 0.9, 0.9])
        k1.metric("Total Tests", f"{total_tests:,}")
        k2.metric("Safe", f"{int(safe_count):,}")
        k3.metric("Not Safe", f"{int(unsafe_count):,}")

        # Filters: date range and result type
        with st.expander("Filters", expanded=False):
            colf1, colf2, colf3 = st.columns([1, 1, 1])
            # date range default to last 30 days
            max_ts = logs_df["timestamp"].max()
            min_ts = logs_df["timestamp"].min()
            if pd.isna(min_ts) or pd.isna(max_ts):
                # fallback to today
                min_def = datetime.now().date()
                max_def = datetime.now().date()
            else:
                min_def = (max_ts - pd.Timedelta(days=30)).date()
                max_def = max_ts.date()

            date_from = colf1.date_input("From", value=min_def)
            date_to = colf1.date_input("To", value=max_def)
            result_filter = colf2.selectbox("Result", ["All", "Safe", "Not Safe", "Unknown"])
            text_search = colf3.text_input("Search (UPC / Ingredient / Competitor)")

        # Apply filters
        df_filtered = logs_df.copy()
        # date filter
        try:
            df_filtered = df_filtered[
                (df_filtered["timestamp"].dt.date >= date_from) &
                (df_filtered["timestamp"].dt.date <= date_to)
            ]
        except Exception:
            # if timestamp parsing failed, ignore date filter
            pass

        # result filter
        if result_filter != "All":
            df_filtered = df_filtered[df_filtered["Result"].astype(str).str.lower() == result_filter.lower()]

        # text search
        if text_search and text_search.strip():
            q = text_search.strip().lower()
            df_filtered = df_filtered[
                df_filtered["UPC"].astype(str).str.lower().str.contains(q, na=False) |
                df_filtered["Ingredient"].astype(str).str.lower().str.contains(q, na=False) |
                df_filtered["Competitor"].astype(str).str.lower().str.contains(q, na=False)
            ]

        # Trend chart (tests per day)
        try:
            trend = df_filtered.dropna(subset=["timestamp"]).groupby(df_filtered["timestamp"].dt.date).size()
            if not trend.empty:
                st.altair_chart(
                    (px.line(x=trend.index, y=trend.values, labels={"x": "Date", "y": "Tests"})
                     .update_layout(margin=dict(l=20, r=20, t=20, b=20))),
                    use_container_width=True
                )
        except Exception:
            pass

        # Prepare Recent Tests table (most recent first)
        to_show = df_filtered.sort_values(by="timestamp", ascending=False).head(20).copy()
        # Format timestamp for display
        if "timestamp" in to_show.columns:
            to_show["timestamp"] = to_show["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

        # Select columns to display
        display_cols = ["timestamp", "UPC", "Ingredient", "Competitor", "Result", "suggestions_text"]
        display_cols = [c for c in display_cols if c in to_show.columns]

        # Build a styled table: color Result cell
        def _result_color(val):
            s = str(val).lower()
            if s == "safe":
                return "background-color: #e6f4ea; color: #0b6623; font-weight:600;"
            if s == "not safe":
                return "background-color: #fdecea; color: #a80000; font-weight:600;"
            return ""

        st.markdown("### Recent Tests")
        try:
            styled = to_show[display_cols].style.applymap(lambda v: _result_color(v) if isinstance(v, str) and v in to_show["Result"].values else "", subset=["Result"])
            # ensure 'Suggestions' column is readable label
            rename_map = {"suggestions_text": "Suggestions", "timestamp": "Date"}
            st.write(styled.rename(columns=rename_map))
        except Exception:
            # fallback: plain table
            st.dataframe(to_show[display_cols].rename(columns={"suggestions_text": "Suggestions", "timestamp": "Date"}), use_container_width=True)

        # Expandable details for each recent test (professional layout)
        st.markdown("### Inspect Results")
        for idx, row in to_show.iterrows():
            ts = row.get("timestamp", "")
            label = f"{ts} — {row.get('Ingredient','')} — {row.get('Result','')}"
            with st.expander(label, expanded=False):
                # left / right columns
                cL, cR = st.columns([2, 1])
                with cL:
                    st.subheader("Test Details")
                    st.write("**UPC:**", row.get("UPC", ""))
                    st.write("**Ingredient:**", row.get("Ingredient", ""))
                    st.write("**Competitor:**", row.get("Competitor", ""))
                    st.write("**Result:**", row.get("Result", ""))
                    if "confidence" in row:
                        st.write("**Confidence:**", row.get("confidence"))
                    # suggestions block (black text, professional typography)
                    sugg = row.get("suggestions", row.get("suggestions_text", ""))
                    if sugg:
                        st.markdown("<div style='color:#111111; font-weight:600; margin-top:8px;'>Suggested Improvements</div>", unsafe_allow_html=True)
                        # show each suggestion on its own line
                        if isinstance(sugg, (list, tuple)):
                            for s in sugg:
                                st.markdown(f"- {s}")
                        else:
                            # if suggestions_text string (pipe-separated), split for readability
                            if isinstance(sugg, str) and " | " in sugg:
                                for s in sugg.split(" | "):
                                    st.markdown(f"- {s.strip()}")
                            else:
                                st.markdown(f"- {sugg}")
                    # show any saved per-criterion details if present
                    if "details" in row and row["details"]:
                        st.markdown("**Per-criterion details:**")
                        try:
                            for d in row["details"]:
                                st.markdown(f"- {d}")
                        except Exception:
                            st.write(row["details"])

                with cR:
                    st.subheader("Raw Log")
                    # show compact JSON-like record for audit
                    clean_row = row.to_dict()
                    # remove big objects if any
                    for k in ["chart", "image", "file"] :
                        if k in clean_row:
                            clean_row.pop(k, None)
                    st.json(clean_row)

        # Downloads: recent view and full logs
        download_col1, download_col2 = st.columns(2)
        try:
            # recent view CSV
            recent_csv = to_show[display_cols].to_csv(index=False).encode("utf-8")
            download_col1.download_button(
                "⬇️ Download shown logs (CSV)",
                data=recent_csv,
                file_name=f"medsafe_shown_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        except Exception:
            download_col1.info("Recent logs download unavailable.")

        try:
            full_csv = logs_df.to_csv(index=False).encode("utf-8")
            download_col2.download_button(
                "⬇️ Download ALL logs (CSV)",
                data=full_csv,
                file_name=f"medsafe_all_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        except Exception:
            download_col2.info("Full logs download unavailable.")

        # Optional admin action: clear logs (dangerous) — show only to admin role
        if (role or "").strip().lower() == "admin":
            with st.expander("Admin actions", expanded=False):
                if st.button("⚠️ Clear all logs (ADMIN ONLY)"):
                    try:
                        log_col.delete_many({})
                        st.success("All logs deleted.")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error("Failed to clear logs: " + str(e))




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
