
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
set_background("bg5.png")
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
        comp_vals[c] = st.number_input(f"{c}", value=0.0, format="%.2f")

    # -------------------------
    # Helper: build competitor DataFrame (kept local to testing flow)
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
    # Compare button: everything below runs only on Testing page
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

            # Run the strict rule-based comparator
            rule_out = predict_rule_based(active_ing, competitor_dict, std_row=std_row)
            result = rule_out["result"]
            details = rule_out["details"]
            confidence = rule_out["confidence"]

            # Show result
            if result.lower() == "safe":
                st.success(f"✅ Prediction → {result} (confidence {confidence:.2f})")
            else:
                st.error(f"❌ Prediction → {result} (confidence {confidence:.2f})")

            # Per-criterion details
            st.markdown("### 🔎 Per-criterion details")
            for d in details:
                if "PASS" in d:
                    st.markdown(f"<div style='color:green'>{d}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='color:red'>{d}</div>", unsafe_allow_html=True)

            # Suggestions (black text as requested)
            suggestions_list = suggestions(competitor_dict)
            if suggestions_list:
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

            # Logging: try Mongo then CSV fallback
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

            # PDF Report (includes suggestions if present)
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
# 📊 PROFESSIONAL DASHBOARD (Fixed + New Features: Alerts, Ranking, Forecast, Ingredient Profile)
# Replace your existing `elif menu == "📊 Dashboard":` block with this.
# =========================================================
elif menu == "📊 Dashboard":
    st.header("📊 Test Results Dashboard")
    st.markdown(
        "<div style='color:#222; font-size:15px;'>Overview of medicine safety comparisons — "
        "KPIs, trends, alerts for unsafe results, competitor ranking, forecasting, and per-ingredient analysis.</div>",
        unsafe_allow_html=True
    )
    st.write("")

    # ---------- 1) LOAD LOGS SAFELY ----------
    try:
        raw_logs = list(log_col.find({}).sort([("_id", -1)]))
    except Exception as e:
        raw_logs = []
        st.error("❌ Unable to load logs from MongoDB: " + str(e))
        st.stop()

    if not raw_logs:
        st.info("No logs found. Run tests from the Testing page to populate logs.")
        st.stop()

    logs_df = pd.DataFrame(raw_logs)

    # ---------- 2) NORMALIZE FIELDS (robust) ----------
    # Ensure timestamp
    logs_df["timestamp"] = pd.to_datetime(logs_df.get("timestamp", pd.NaT), errors="coerce")
    # Normalize text fields (use get fallback)
    logs_df["UPC"] = logs_df.get("UPC", logs_df.get("upc", "")).astype(str)
    logs_df["Ingredient"] = logs_df.get("Ingredient", logs_df.get("ingredient", "")).astype(str)
    logs_df["Competitor"] = logs_df.get("Competitor", logs_df.get("competitor", "")).astype(str)
    logs_df["Result"] = logs_df.get("Result", logs_df.get("result", "Unknown")).astype(str)

    # Normalize suggestions safely (fixes previous AttributeError)
    def _norm_sugg_cell(cell):
        if cell is None:
            return ""
        if isinstance(cell, (list, tuple)):
            return " | ".join(map(str, cell))
        if isinstance(cell, dict):
            # flatten dict values
            return " | ".join(f"{k}: {v}" for k, v in cell.items())
        # any other scalar
        return str(cell)

    if "suggestions" in logs_df.columns:
        logs_df["Suggestions"] = logs_df["suggestions"].apply(_norm_sugg_cell)
    else:
        logs_df["Suggestions"] = ""

    # Normalize details if present
    if "details" not in logs_df.columns:
        logs_df["details"] = [[] for _ in range(len(logs_df))]
    else:
        # ensure lists
        def _norm_details(d):
            if d is None:
                return []
            if isinstance(d, list):
                return d
            if isinstance(d, str):
                # try split by '|' or newline if accidentally stored as string
                if " | " in d:
                    return [x.strip() for x in d.split(" | ") if x.strip()]
                return [d]
            if isinstance(d, dict):
                return [f"{k}: {v}" for k, v in d.items()]
            return [str(d)]
        logs_df["details"] = logs_df["details"].apply(_norm_details)

    # ---------- 3) KPIs ----------
    total_tests = len(logs_df)
    safe_count = logs_df["Result"].str.lower().eq("safe").sum()
    not_safe_count = logs_df["Result"].str.lower().eq("not safe").sum()
    unknown_count = total_tests - safe_count - not_safe_count

    k1, k2, k3, k4 = st.columns([1.2, 0.9, 0.9, 0.9])
    k1.metric("Total Tests", f"{total_tests:,}")
    k2.metric("Safe", f"{int(safe_count):,}", delta=f"{(safe_count/total_tests*100):.1f}%")
    k3.metric("Not Safe", f"{int(not_safe_count):,}", delta=f"{(not_safe_count/total_tests*100):.1f}%")
    k4.metric("Unknown", f"{int(unknown_count):,}")

    st.markdown("---")

    # ---------- 4) FILTER BAR ----------
    with st.expander("🔎 Filters (date / result / search)", expanded=False):
        cf1, cf2, cf3 = st.columns(3)
        min_ts = logs_df["timestamp"].min()
        max_ts = logs_df["timestamp"].max()
        if pd.isna(min_ts):
            min_ts = datetime.now()
        if pd.isna(max_ts):
            max_ts = datetime.now()

        date_from = cf1.date_input("From", min_ts.date())
        date_to = cf1.date_input("To", max_ts.date())
        result_filter = cf2.selectbox("Result", ["All", "Safe", "Not Safe", "Unknown"])
        search_txt = cf3.text_input("Search (UPC / Ingredient / Competitor)")

    # Apply filters to a working copy
    df_filtered = logs_df.copy()
    try:
        df_filtered = df_filtered[
            (df_filtered["timestamp"].dt.date >= date_from) &
            (df_filtered["timestamp"].dt.date <= date_to)
        ]
    except Exception:
        # timestamps may be NaT — ignore date filtering in that case
        pass

    if result_filter != "All":
        df_filtered = df_filtered[df_filtered["Result"].str.lower() == result_filter.lower()]

    if search_txt and search_txt.strip():
        q = search_txt.strip().lower()
        df_filtered = df_filtered[
            df_filtered["UPC"].str.lower().str.contains(q, na=False) |
            df_filtered["Ingredient"].str.lower().str.contains(q, na=False) |
            df_filtered["Competitor"].str.lower().str.contains(q, na=False)
        ]

    st.markdown("---")

    # ---------- 5) TREND + SIMPLE FORECAST ----------
    st.markdown("### 📈 Tests Trend & 7-day Forecast")
    try:
        # counts per day
        daily = df_filtered.dropna(subset=["timestamp"]).groupby(df_filtered["timestamp"].dt.date).size().rename("count")
        if daily.empty:
            st.info("Not enough dated data for trend.")
        else:
            fig_trend = px.line(x=daily.index, y=daily.values, labels={"x":"Date","y":"Tests"}, title="Tests per day")
            st.plotly_chart(fig_trend, use_container_width=True)

            # Simple linear forecast (fit line to daily counts)
            try:
                # numeric x
                x = np.arange(len(daily))
                y = np.array(daily.values)
                if len(x) >= 3:
                    # fit degree-1 poly
                    p = np.polyfit(x, y, 1)
                    next_x = np.arange(len(daily), len(daily)+7)
                    pred = np.polyval(p, next_x).clip(min=0).round().astype(int)

                    future_dates = [ (pd.to_datetime(daily.index[-1]) + pd.Timedelta(days=i)).date() for i in range(1,8) ]
                    forecast_df = pd.DataFrame({"date": future_dates, "predicted_tests": pred})
                    st.markdown("**7-day simple forecast (linear trend)**")
                    st.table(forecast_df)
                else:
                    st.info("Need at least 3 days of data for a basic forecast.")
            except Exception:
                st.info("Forecasting failed (insufficient data).")
    except Exception:
        st.info("Trend chart unavailable.")

    st.markdown("---")

    # ---------- 6) UNSAFE ALERTS PANEL (Feature 2) ----------
    st.markdown("### 🚨 Recent Unsafe Alerts")
    try:
        recent_unsafe = df_filtered[df_filtered["Result"].str.lower() == "not safe"].sort_values(by="timestamp", ascending=False).head(10)
        if recent_unsafe.empty:
            st.info("No recent unsafe cases in the selected filter range.")
        else:
            for idx, r in recent_unsafe.iterrows():
                with st.container():
                    row_time = r.get("timestamp")
                    if pd.isna(row_time):
                        row_time_str = "Unknown date"
                    else:
                        row_time_str = pd.to_datetime(row_time).strftime("%Y-%m-%d %H:%M:%S")
                    st.markdown(f"**{row_time_str}** — **{r.get('Ingredient','')}** — {r.get('Competitor','')}")
                    # show the failed criteria (parse details)
                    failed = []
                    for d in r.get("details", []):
                        if isinstance(d, str) and "FAIL" in d.upper():
                            failed.append(d)
                    if failed:
                        for fline in failed:
                            st.markdown(f"<div style='color:#a80000; font-weight:600'>{fline}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='color:#a80000; font-weight:600'>No per-criterion FAIL lines found in log.</div>", unsafe_allow_html=True)
                    # show suggestions (black)
                    sugg = r.get("Suggestions", "")
                    if sugg:
                        st.markdown(f"<div style='color:#111111; margin-top:6px;'>Suggested Improvements: {sugg}</div>", unsafe_allow_html=True)
                    st.markdown("---")
    except Exception:
        st.info("Unable to build Unsafe Alerts.")

    # ---------- 7) COMPETITOR RANKING (Feature 3) ----------
    st.markdown("### 🏆 Competitor Ranking (by tests & safety rate)")
    try:
        comp_grp = df_filtered.groupby("Competitor").agg(
            tests=("Competitor", "count"),
            safe_count=("Result", lambda s: s.str.lower().eq("safe").sum()),
        ).reset_index()
        if not comp_grp.empty:
            comp_grp["safe_rate"] = (comp_grp["safe_count"] / comp_grp["tests"] * 100).round(1)
            comp_rank = comp_grp.sort_values(["safe_rate", "tests"], ascending=[False, False]).head(20)
            st.dataframe(comp_rank.rename(columns={"tests":"Tests","safe_count":"Safe Count","safe_rate":"Safety (%)"}), use_container_width=True)

            # small bar chart of top 10 by safety rate (only competitors with >1 test)
            chart_df = comp_rank[comp_rank["tests"]>0].head(10)
            fig_comp = px.bar(chart_df, x="Competitor", y="safe_rate", labels={"safe_rate":"Safety (%)"}, title="Top Competitors by Safety Rate")
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.info("Not enough competitor data.")
    except Exception:
        st.info("Competitor ranking unavailable.")

    st.markdown("---")

    # ---------- 8) INGREDIENT PROFILE (Feature 5) ----------
    st.markdown("### 🔬 Ingredient Profile")
    try:
        ingredients = sorted(logs_df["Ingredient"].dropna().unique().tolist())
        sel_ing = st.selectbox("Select ingredient", options=["(all)"] + ingredients, index=0)
        if sel_ing == "(all)":
            ing_df = logs_df.copy()
        else:
            ing_df = logs_df[logs_df["Ingredient"] == sel_ing].copy()

        st.write(f"Showing {len(ing_df)} tests for ingredient: **{sel_ing}**")

        # Pass/fail by criterion: parse 'details' lines that contain 'PASS'/'FAIL'
        # Build counts per criterion
        criterion_counts = {}
        for details in ing_df["details"].tolist():
            for d in details:
                if not isinstance(d, str):
                    continue
                # attempt to parse "Criterion: value ... → PASS" or "→ FAIL"
                upper = d.upper()
                if "→" in d:
                    try:
                        left, right = d.split("→", 1)
                        outcome = right.strip().upper()
                        # criterion name is left before colon
                        crit = left.split(":", 1)[0].strip()
                    except Exception:
                        crit = d
                        outcome = "UNKNOWN"
                else:
                    # fallback detection
                    crit = d.split(":",1)[0].strip()
                    outcome = "PASS" if "PASS" in upper else ("FAIL" if "FAIL" in upper else "UNKNOWN")
                if crit not in criterion_counts:
                    criterion_counts[crit] = {"pass":0, "fail":0, "total":0}
                if "PASS" in outcome:
                    criterion_counts[crit]["pass"] += 1
                elif "FAIL" in outcome:
                    criterion_counts[crit]["fail"] += 1
                else:
                    # unknown — count as total only
                    pass
                criterion_counts[crit]["total"] += 1

        if criterion_counts:
            crit_df = pd.DataFrame([
                {"Criterion": k, "Pass": v["pass"], "Fail": v["fail"], "Total": v["total"],
                 "PassRate": (v["pass"]/v["total"]*100 if v["total"]>0 else 0)}
                for k, v in criterion_counts.items()
            ]).sort_values("PassRate", ascending=False)

            st.dataframe(crit_df, use_container_width=True)

            # bar chart of pass rate
            fig_ing = px.bar(crit_df, x="Criterion", y="PassRate", labels={"PassRate":"Pass Rate (%)"}, title="Per-criterion pass rate")
            st.plotly_chart(fig_ing, use_container_width=True)
        else:
            st.info("No per-criterion detail lines found for this ingredient.")

    except Exception:
        st.info("Ingredient profile unavailable.")

    st.markdown("---")

    # ---------- 9) PAGINATED TABLE + DOWNLOADS ----------
    st.markdown("### 📋 Recent Results (paginated)")
    try:
        page_size = st.selectbox("Rows per page", [10, 20, 50], index=1)
        total_pages = max(1, (len(df_filtered) + page_size - 1) // page_size)
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
        start = (page-1) * page_size
        end = start + page_size
        df_page = df_filtered.iloc[start:end].copy()

        display_cols = ["timestamp", "UPC", "Ingredient", "Competitor", "Result", "Suggestions"]
        display_cols = [c for c in display_cols if c in df_page.columns]
        if "timestamp" in df_page.columns:
            df_page["timestamp"] = df_page["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(df_page[display_cols].rename(columns={"timestamp":"Date","Suggestions":"Suggestions"}), use_container_width=True)

        c1, c2 = st.columns(2)
        try:
            csv_page = df_page[display_cols].to_csv(index=False).encode("utf-8")
            c1.download_button("⬇️ Download visible page (CSV)", csv_page, file_name="visible_logs.csv", mime="text/csv")
        except Exception:
            pass
        try:
            full_csv = logs_df.to_csv(index=False).encode("utf-8")
            c2.download_button("⬇️ Download ALL logs (CSV)", full_csv, file_name="all_logs.csv", mime="text/csv")
        except Exception:
            pass
    except Exception:
        st.info("Unable to render paginated table.")

    # ---------- 10) ADMIN: CLEAR LOGS ----------
    if (role or "").strip().lower() == "admin":
        with st.expander("⚠️ Admin Actions", expanded=False):
            if st.button("Delete ALL Logs (ADMIN ONLY)"):
                try:
                    log_col.delete_many({})
                    st.success("All logs deleted. Reloading...")
                    st.experimental_rerun()
                except Exception as e:
                    st.error("Failed to clear logs: " + str(e))

# =========================================================
# End of Dashboard replacement
# =========================================================





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
            for k in [
                "Item Name", "Category", "UPC", "Quantity in Stock",
                "Expiry Period (Months)", "Storage Temperature (C)", "Safe/Not Safe"
            ]:
                if k not in d:
                    d[k] = None
        return pd.DataFrame(docs)

    meds = load_medicines()
    cons = load_consumables()

    tab1, tab2 = st.tabs(["💊 Medicines", "🛠 Consumables"])

    # ------------------------------
    # Helper: apply filters to a DF (Medicines)
    # ------------------------------
    def apply_medicine_filters(df):
        with st.expander("🔎 Filters (Medicines)", expanded=False):
            col1, col2, col3 = st.columns(3)
            f_upc = col1.text_input("Filter UPC", value="", key="med_f_upc")
            f_ing = col1.text_input("Filter Ingredient", value="", key="med_f_ing")
            f_batch = col2.text_input("Filter Batch", value="", key="med_f_batch")
            f_mf = col2.text_input("Filter Manufacturer", value="", key="med_f_mf")
            f_low_stock = col3.number_input("Low stock threshold (≤)", min_value=0, value=0, step=1, key="med_f_low_stock")

            df_filtered = df.copy()
            if f_upc.strip():
                df_filtered = df_filtered[df_filtered["UPC"].astype(str).str.contains(f_upc.strip(), case=False, na=False)]
            if f_ing.strip():
                df_filtered = df_filtered[df_filtered["Ingredient"].astype(str).str.contains(f_ing.strip(), case=False, na=False)]
            if f_batch.strip():
                df_filtered = df_filtered[df_filtered["Batch"].astype(str).str.contains(f_batch.strip(), case=False, na=False)]
            if f_mf.strip():
                df_filtered = df_filtered[df_filtered["Manufacturer"].astype(str).str.contains(f_mf.strip(), case=False, na=False)]
            if f_low_stock > 0 and "Stock" in df_filtered.columns:
                df_filtered["Stock"] = pd.to_numeric(df_filtered["Stock"], errors="coerce").fillna(0).astype(int)
                df_filtered = df_filtered[df_filtered["Stock"] <= int(f_low_stock)]
        return df_filtered

    # ------------------------------
    # Helper: apply filters to a DF (Consumables) — correctly indented
    # ------------------------------
    def apply_consumable_filters(df):
        with st.expander("🔎 Filters (Consumables)", expanded=False):
            col1, col2, col3 = st.columns(3)
            f_name = col1.text_input("Filter Item Name", value="", key="cons_f_name")
            f_cat = col1.text_input("Filter Category", value="", key="cons_f_cat")
            f_upc = col2.text_input("Filter UPC", value="", key="cons_f_upc")
            f_safe = col3.selectbox("Safe / Not Safe", ["All", "Safe", "Not Safe"], key="cons_f_safe")
            f_low_qty = col3.number_input("Low qty threshold (≤)", min_value=0, value=0, step=1, key="cons_f_low_qty")

            df_filtered = df.copy()
            if f_name.strip():
                df_filtered = df_filtered[df_filtered["Item Name"].astype(str).str.contains(f_name.strip(), case=False, na=False)]
            if f_cat.strip():
                df_filtered = df_filtered[df_filtered["Category"].astype(str).str.contains(f_cat.strip(), case=False, na=False)]
            if f_upc.strip():
                df_filtered = df_filtered[df_filtered["UPC"].astype(str).str.contains(f_upc.strip(), case=False, na=False)]
            if f_safe != "All":
                df_filtered = df_filtered[df_filtered["Safe/Not Safe"] == f_safe]
            if f_low_qty > 0 and "Quantity in Stock" in df_filtered.columns:
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

        # Add / Update form (Medicines)
        st.markdown("### ➕ Add / Update Medicine")
        with st.form("add_med_form"):
            colA, colB, colC = st.columns(3)
            upc = colA.text_input("UPC", key="add_med_upc")
            ing = colA.text_input("Ingredient", key="add_med_ing")
            mf = colB.text_input("Manufacturer", key="add_med_mf")
            batch = colB.text_input("Batch", key="add_med_batch")
            stock = colC.number_input("Stock", min_value=0, value=1, key="add_med_stock")
            expiry = colC.date_input("Expiry Date", key="add_med_expiry")
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
            sel_idx = st.selectbox("Select medicine", options=list(range(len(labels))), format_func=lambda i: labels[i], key="med_select")
            real_idx = idx_list[sel_idx]
            rec = meds_filtered.loc[real_idx]
            sel_id = rec["_id"]

            col1, col2, col3 = st.columns(3)
            new_ing = col1.text_input("Ingredient", rec.get("Ingredient", ""), key="edit_med_ing")
            new_upc = col1.text_input("UPC", rec.get("UPC", ""), key="edit_med_upc")
            new_mf = col2.text_input("Manufacturer", rec.get("Manufacturer", ""), key="edit_med_mf")
            new_batch = col2.text_input("Batch", rec.get("Batch", ""), key="edit_med_batch")
            new_stock = col3.number_input("Stock", min_value=0, value=int(rec.get("Stock") or 0), key="edit_med_stock")
            try:
                old_exp = pd.to_datetime(rec.get("Expiry")).date()
            except:
                old_exp = datetime.today().date()
            new_exp = col3.date_input("Expiry", old_exp, key="edit_med_expiry")

            if st.button("Save Medicine Changes", key="save_med_changes"):
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

            if st.button("Delete Medicine", key="delete_med"):
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
        with st.form("add_cons_form"):
            colA, colB = st.columns(2)
            name = colA.text_input("Item Name", key="add_cons_name")
            category = colA.text_input("Category", key="add_cons_cat")
            upc = colB.text_input("UPC", key="add_cons_upc")
            qty = colB.number_input("Quantity", min_value=0, value=1, key="add_cons_qty")
            expiry_m = colA.number_input("Expiry (Months)", min_value=0, value=12, key="add_cons_expiry")
            storage = colB.number_input("Storage Temp (°C)", value=25, key="add_cons_storage")
            safe_flag = colA.selectbox("Safe / Not Safe", ["Safe", "Not Safe"], key="add_cons_safe")
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

            labels = [
                f"{r.get('Item Name','(no name)')} | UPC:{r.get('UPC','')} | Qty:{int(r.get('Quantity in Stock') or 0)}"
                for _, r in cons_filtered.iterrows()
            ]
            idx_list = list(cons_filtered.index)
            sel_idx = st.selectbox("Select consumable", options=list(range(len(labels))), format_func=lambda i: labels[i], key="cons_select")
            real_idx = idx_list[sel_idx]
            rec = cons_filtered.loc[real_idx]
            sel_id = rec["_id"]

            col1, col2 = st.columns(2)
            new_item = col1.text_input("Item Name", rec.get("Item Name", ""), key="edit_cons_name")
            new_cat = col1.text_input("Category", rec.get("Category", ""), key="edit_cons_cat")
            new_upc = col2.text_input("UPC", rec.get("UPC", ""), key="edit_cons_upc")
            new_qty = col2.number_input("Quantity", min_value=0, value=int(rec.get("Quantity in Stock") or 0), key="edit_cons_qty")
            new_exp = col1.number_input("Expiry (Months)", min_value=0, value=int(rec.get("Expiry Period (Months)") or 0), key="edit_cons_expiry")
            new_safe = col2.selectbox("Safe / Not Safe", ["Safe", "Not Safe"], index=0 if rec.get("Safe/Not Safe","Safe")=="Safe" else 1, key="edit_cons_safe")

            if st.button("Save Consumable Changes", key="save_cons_changes"):
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

            if st.button("Delete Consumable", key="delete_cons"):
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
