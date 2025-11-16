import streamlit as st
import base64
import os

# ====================================
# DISABLE BACKGROUND ON LOGIN PAGE
# ====================================
def disable_all_background_for_login():
    """Remove all backgrounds & padding ONLY for login screen."""
    st.markdown("""
        <style>
            .stApp, .block-container {
                background: transparent !important;
                padding: 0 !important;
                margin: 0 !important;
            }
            header, footer, [data-testid="stToolbar"],
            [data-testid="stDecoration"], [data-testid="stHeader"] {
                display: none !important;
            }
        </style>
    """, unsafe_allow_html=True)

# ====================================
# THEME SETUP
# ====================================
def apply_theme():
    if "theme_choice" not in st.session_state:
        st.session_state.theme_choice = "Light"
    if "custom_theme" not in st.session_state:
        st.session_state.custom_theme = {
            "text_color": "#000000",
            "metric_bg": "#f5f5f5",
            "button_text": "#000000",
            "button_bg": "#e0e0e0",
            "header_color": "#000000",
        }

# ====================================
# APPLY LAYOUT + SIDEBAR SETTINGS
# ====================================
def apply_layout_styles():
    THEMES = {
        "Light": {
            "text_color": "#000000",
            "metric_bg": "#ffffff",
            "button_text": "#000000",
            "button_bg": "#eaeaea",
            "header_color": "#000000",
        },
        "Dark": {
            "text_color": "#ffffff",
            "metric_bg": "#333333",
            "button_text": "#ffffff",
            "button_bg": "#444444",
            "header_color": "#ffffff",
        }
    }

    st.sidebar.markdown("### 🎨 Theme Settings")

    theme_choice = st.sidebar.radio(
        "Choose Theme",
        ["Light", "Dark", "Custom"],
        index=["Light", "Dark", "Custom"].index(st.session_state.theme_choice)
    )
    st.session_state.theme_choice = theme_choice

    if theme_choice in THEMES:
        THEME = THEMES[theme_choice]
    else:
        custom = st.session_state.custom_theme
        custom["text_color"]   = st.sidebar.color_picker("Text Color", custom["text_color"])
        custom["metric_bg"]    = st.sidebar.color_picker("KPI Background", custom["metric_bg"])
        custom["button_text"]  = st.sidebar.color_picker("Button Text", custom["button_text"])
        custom["button_bg"]    = st.sidebar.color_picker("Button BG", custom["button_bg"])
        custom["header_color"] = st.sidebar.color_picker("Header Color", custom["header_color"])
        st.session_state.custom_theme = custom
        THEME = custom

    st.markdown(f"""
        <style>
            body, html {{
                color: {THEME['text_color']} !important;
            }}
            .stMetric {{
                background: {THEME['metric_bg']} !important;
                padding: 12px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .stButton > button {{
                background: {THEME['button_bg']} !important;
                color: {THEME['button_text']} !important;
                font-weight: 600;
                border-radius: 8px;
                padding: 0.6em;
            }}
            h1, h2, h3, h4, h5 {{
                color: {THEME['header_color']} !important;
            }}
        </style>
    """, unsafe_allow_html=True)

# ====================================
# GLOBAL CSS — SAFE FOR DASHBOARD
# ====================================
def apply_global_css():
    st.markdown("""
        <style>
            .block-container {
                padding-top: 0.5rem !important;
            }
            [data-testid="stDecoration"] {
                display: none !important;
            }
            header[data-testid="stHeader"] {
                display: none !important;
                height: 0px !important;
            }
        </style>
    """, unsafe_allow_html=True)

# ====================================
# SAFE BACKGROUND (login unaffected)
# ====================================
def set_background(image_file):
    if not os.path.exists(image_file):
        return
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
            body {{
                background-image: url("data:image/png;base64,{encoded}");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
                background-position: center;
            }}
            .stApp {{
                background: transparent !important;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ====================================
# SHOW LOGO — CENTER TOP
# ====================================
def show_logo(logo_file):
    if not os.path.exists(logo_file):
        return
    with open(logo_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <div style="width:100%; text-align:center; margin:10px 0;">
            <img src="data:image/png;base64,{encoded}" style="width:140px;">
        </div>
        """,
        unsafe_allow_html=True
    )

# ====================================
# FIX: RESTORE LAYOUT AFTER LOGIN
# ====================================
def restore_default_layout():
    """Restores normal Streamlit layout after login."""
    st.markdown("""
        <style>
            .block-container {
                padding-top: 1rem !important;
                margin-top: 0 !important;
            }
            .stApp {
                padding: 0 !important;
                margin: 0 !important;
            }
            /* Ensure sidebar displays normally */
            section[data-testid="stSidebar"] {
                display: block !important;
            }
        </style>
    """, unsafe_allow_html=True)
