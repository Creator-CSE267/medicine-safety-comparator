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


# ===============================
# GLOBAL CSS — SAFE FOR LOGIN PAGE
# ===============================
def apply_global_css():
    st.markdown("""
        <style>

        /* Remove any default padding that creates blank area */
        .block-container {
            padding-top: 0rem !important;
        }

        /* Remove Streamlit top blank spacer */
        div[data-testid="stDecoration"] {
            display: none !important;
        }

        div[data-testid="stToolbar"] {
            display: none !important;
        }

        header[data-testid="stHeader"] {
            height: 0px !important;
            display: none !important;
        }

        /* Login box centering */
        .login-container {
            max-width: 420px !important;
            margin: auto !important;
            margin-top: 60px !important;
        }

        .main-title {
            font-size: 28px;
            font-weight: bold;
            color: #ffffff;
        }

        </style>
    """, unsafe_allow_html=True)



# ===============================
# BACKGROUND (ONLY AFTER LOGIN)
# ===============================
def set_background(image_file):
    """Set background image AFTER LOGIN ONLY."""
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
        </style>
        """,
        unsafe_allow_html=True
    )


# ===============================
# LOGO — CENTERED FOR DASHBOARD ONLY
# ===============================
def show_logo(logo_file):
    """Display logo centered without adding unwanted white box."""
    if os.path.exists(logo_file):
        import base64
        with open(logo_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        st.markdown(
            f"""
            <style>
                .logo-container {{
                    width: 100%;
                    display: flex;
                    justify-content: center;
                    margin-top: 20px;
                    margin-bottom: 10px;
                }}
                .logo-container img {{
                    width: 140px;
                }}
            </style>

            <div class="logo-container">
                <img src="data:image/png;base64,{encoded}">
            </div>
            """,
            unsafe_allow_html=True
        )
