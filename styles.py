import streamlit as st
import base64
import os

# ====================================================
#  LOGIN PAGE — ONLY REMOVE BACKGROUND + SPACING HERE
# ====================================================
def disable_all_background_for_login():
    """Used ONLY on login screen."""
    st.markdown("""
        <style>
            .stApp, .block-container {
                background: transparent !important;
                padding: 0 !important;
                margin: 0 !important;
            }

            /* Hide header ONLY FOR LOGIN */
            header, footer,
            [data-testid="stToolbar"],
            [data-testid="stDecoration"] {
                display: none !important;
            }
        </style>
    """, unsafe_allow_html=True)


# ====================================================
#  RESTORE STREAMLIT DEFAULT LAYOUT AFTER LOGIN
# ====================================================
def restore_default_layout():
    """Re-enable normal layout so sidebar appears."""
    st.markdown("""
        <style>
            header, [data-testid="stHeader"] {
                display: block !important;
                visibility: visible !important;
                height: auto !important;
            }

            .block-container {
                padding-top: 1rem !important;
                margin-top: 0 !important;
            }
        </style>
    """, unsafe_allow_html=True)


# ====================================================
# THEME INITIALIZER
# ====================================================
def apply_theme():
    if "theme_choice" not in st.session_state:
        st.session_state.theme_choice = "Light"
    if "custom_theme" not in st.session_state:
        st.session_state.custom_theme = {
            "text_color": "#000",
            "metric_bg": "#f5f5f5",
            "button_text": "#000",
            "button_bg": "#e0e0e0",
            "header_color": "#000",
        }


# ====================================================
# SIDEBAR THEME CONTROLS
# ====================================================
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

    choice = st.sidebar.radio(
        "Choose Theme",
        ["Light", "Dark", "Custom"],
        index=["Light", "Dark", "Custom"].index(st.session_state.theme_choice)
    )
    st.session_state.theme_choice = choice

    if choice in THEMES:
        THEME = THEMES[choice]
    else:
        custom = st.session_state.custom_theme
        custom["text_color"] = st.sidebar.color_picker("Text Color", custom["text_color"])
        custom["metric_bg"] = st.sidebar.color_picker("KPI Card Background", custom["metric_bg"])
        custom["button_text"] = st.sidebar.color_picker("Button Text Color", custom["button_text"])
        custom["button_bg"] = st.sidebar.color_picker("Button BG", custom["button_bg"])
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
            }}
            .stButton > button {{
                background: {THEME['button_bg']} !important;
                color: {THEME['button_text']} !important;
                border-radius: 8px;
                font-weight: 600;
            }}
            h1, h2, h3, h4 {{
                color: {THEME['header_color']} !important;
            }}
        </style>
    """, unsafe_allow_html=True)


# ====================================================
# BACKGROUND (SAFE)
# ====================================================
def set_background(image_file):
    if not os.path.exists(image_file):
        return

    encoded = base64.b64encode(open(image_file, "rb").read()).decode()
    st.markdown(f"""
        <style>
            body {{
                background-image: url("data:image/png;base64,{encoded}");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
        </style>
    """, unsafe_allow_html=True)


# ====================================================
# LOGO (SAFE)
# ====================================================
def show_logo(logo_file):
    if not os.path.exists(logo_file):
        return

    encoded = base64.b64encode(open(logo_file, "rb").read()).decode()
    st.markdown(f"""
        <div style="text-align:center; margin: 15px 0;">
            <img src="data:image/png;base64,{encoded}" width="130">
        </div>
    """, unsafe_allow_html=True)


# ====================================
# RESTORE FULL LAYOUT AFTER LOGIN
# ====================================
def restore_default_layout():
    """
    Restores normal Streamlit padding/layout AFTER login.
    Removes ONLY the login-specific overrides.
    Safe for dashboard, sidebar, KPI, tabs, charts.
    """
    st.markdown("""
        <style>
            /* Restore Streamlit container spacing */
            .block-container {
                padding-top: 1rem !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }

            /* Restore header (if needed later) */
            header[data-testid="stHeader"] {
                display: block !important;
                height: auto !important;
            }

            /* Restore toolbar spacing */
            div[data-testid="stToolbar"] {
                display: flex !important;
            }

            /* Restore safe margins */
            .stApp {
                padding-top: 0 !important;
                margin-top: 0 !important;
                background: transparent !important;
            }
        </style>
    """, unsafe_allow_html=True)


