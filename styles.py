# styles.py
import streamlit as st
import base64
from PIL import Image
import os

# ===============================
# Theme & Layout
# ===============================
def apply_theme():
    """Initialize session state theme values."""
    if "theme_choice" not in st.session_state:
        st.session_state.theme_choice = "Light"
    if "custom_theme" not in st.session_state:
        st.session_state.custom_theme = {
            "text_color": "#000000",
            "metric_bg": "#f0f0f0",
            "button_text": "#000000",
            "button_bg": "#e0e0e0",
            "header_color": "#000000",
        }

def apply_layout_styles():
    """Sidebar theme selector + apply chosen theme CSS."""
    THEMES = {
        "Light": {
            "text_color": "#000000",
            "metric_bg": "#f9f9f9",
            "button_text": "#000000",
            "button_bg": "#e0e0e0",
            "header_color": "#000000",
        },
        "Dark": {
            "text_color": "#FFFFFF",
            "metric_bg": "#333333",
            "button_text": "#FFFFFF",
            "button_bg": "#444444",
            "header_color": "#FFFFFF",
        }
    }

    # Sidebar theme selector
    st.sidebar.header("🎨 Theme Settings")
    theme_choice = st.sidebar.radio(
        "Choose Theme",
        ["Light", "Dark", "Custom"],
        index=["Light", "Dark", "Custom"].index(st.session_state.theme_choice)
    )
    st.session_state.theme_choice = theme_choice

    # Apply chosen theme
    if theme_choice in THEMES:
        THEME = THEMES[theme_choice]
    else:
        custom = st.session_state.custom_theme
        custom["text_color"]   = st.sidebar.color_picker("Text Color", custom["text_color"])
        custom["metric_bg"]    = st.sidebar.color_picker("KPI Card Background", custom["metric_bg"])
        custom["button_text"]  = st.sidebar.color_picker("Button Text Color", custom["button_text"])
        custom["button_bg"]    = st.sidebar.color_picker("Button Background", custom["button_bg"])
        custom["header_color"] = st.sidebar.color_picker("Header Color", custom["header_color"])
        st.session_state.custom_theme = custom
        THEME = custom

    # Inject CSS
    st.markdown(f"""
        <style>
            html, body, [class*="css"] {{
                color: {THEME['text_color']} !important;
            }}
            .stMetric {{
                background: {THEME['metric_bg']};
                padding: 15px;
                border-radius: 12px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.08);
                color: {THEME['text_color']} !important;
            }}
            .stButton>button {{
                width: 100%;
                margin-bottom: 10px;
                border-radius: 10px;
                height: 3em;
                font-weight: bold;
                color: {THEME['button_text']} !important;
                background-color: {THEME['button_bg']} !important;
            }}
            h1, h2, h3, h4, h5, h6 {{
                color: {THEME['header_color']} !important;
            }}
        </style>
    """, unsafe_allow_html=True)

def apply_global_css():
    """Extra global CSS for dashboard/inventory cards."""
    st.markdown("""
        <style>
            .main-title {
                font-size: 28px;
                font-weight: bold;
                color: #2E86C1;
                margin-bottom: 20px;
            }
            .section-header {
                font-size: 20px;
                font-weight: bold;
                margin-top: 25px;
                margin-bottom: 10px;
                color: #34495E;
            }
            .card {
                background: #f9f9f9;
                padding: 15px;
                border-radius: 12px;
                margin-bottom: 15px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
        </style>
    """, unsafe_allow_html=True)

# ===============================
# Background & Logo
# ===============================
def set_background(image_file):
    """Set background image for app."""
    if not os.path.exists(image_file):
        return
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            background-attachment: fixed;
        }}
        .block-container {{
            background: transparent !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def show_logo(logo_file):
    """Display logo centered if exists."""
    if os.path.exists(logo_file):
        import base64
        with open(logo_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        # Inject only one centered image
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; margin-bottom: 20px;">
                <img src="data:image/png;base64,{encoded}" width="250">
            </div>
            """,
            unsafe_allow_html=True
        )

