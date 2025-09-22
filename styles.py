import streamlit as st

# -------------------------------
# 🎨 Theme + CSS Styling
# -------------------------------
def apply_theme():
    # Initialize session state for theme persistence
    if "theme_choice" not in st.session_state:
        st.session_state.theme_choice = "Light"   # Default theme
    if "custom_theme" not in st.session_state:
        st.session_state.custom_theme = {
            "text_color": "#000000",
            "metric_bg": "#f0f0f0",
            "button_text": "#000000",
            "button_bg": "#e0e0e0",
            "header_color": "#000000",
        }

    # Preset themes
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

    # Apply global CSS
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


# -------------------------------
# 📐 Layout Styling
# -------------------------------
def apply_layout_styles():
    st.markdown("""
        <style>
        .main-title {
            text-align: center;
            font-size: 2.2em;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .section-header {
            font-size: 1.4em;
            font-weight: 600;
            margin-top: 30px;
            margin-bottom: 15px;
            border-left: 5px solid #2E86C1;
            padding-left: 10px;
        }
        .card {
            background: #ffffff20;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)
