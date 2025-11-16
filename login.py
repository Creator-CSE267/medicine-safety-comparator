import streamlit as st
from user_database import get_user, verify_password
from password_reset import password_reset
from datetime import datetime


# ---------------------------------------------------
# ROUTER (controls login → reset password → return role)
# ---------------------------------------------------
def login_router():

    # Initialize required session states
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if "go_reset_password" not in st.session_state:
        st.session_state["go_reset_password"] = False

    # 1️⃣ RESET PASSWORD PAGE
    if st.session_state["go_reset_password"]:
        username = st.session_state.get("reset_user", "")
        password_reset(username)
        return None, None

    # 2️⃣ ALREADY LOGGED-IN USER
    if st.session_state["authenticated"]:
        return (
            st.session_state["username"],
            st.session_state["role"]
        )

    # 3️⃣ OTHERWISE, SHOW LOGIN PAGE
    return login_page()



# ---------------------------------------------------
# PROFESSIONAL LOGIN PAGE
# ---------------------------------------------------
def login_page():

    st.markdown("""
        <style>
            /* REMOVE ALL DEFAULT TOP MARGINS / PADDING */
            .block-container {
                padding-top: 0rem !important;
                margin-top: 0rem !important;
            }
            header, footer {
                visibility: hidden !important;
            }

            /* FULLSCREEN CENTERING */
            .full-page {
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                height: 100vh;
                width: 100%;
            }

            /* LOGIN CARD */
            .login-card {
                width: 380px;
                padding: 35px;
                background: #1A1C23;
                border-radius: 18px;
                box-shadow: 0px 4px 25px rgba(0,0,0,0.35);
                text-align: center;
            }

            .login-title {
                font-size: 28px;
                font-weight: 800;
                color: #2E86C1;
                margin-top: 12px;
                margin-bottom: 25px;
            }

            .login-input input {
                background: #2A2C33 !important;
                color: white !important;
                border-radius: 10px !important;
                border: 1px solid #2E86C1 !important;
                height: 45px;
            }

            .login-btn, .reset-btn {
                width: 48%;
                border-radius: 8px;
                padding: 10px;
                font-size: 15px;
                font-weight: 600;
            }

        </style>
        <div class='full-page'>
            <div class='login-card'>
    """, unsafe_allow_html=True)

    # Logo (centered)
    st.image("logo.png", width=120)

    # Title
    st.markdown("<div class='login-title'>MedSafe Login</div>", unsafe_allow_html=True)

    # Inputs
    username = st.text_input("Username", placeholder="Enter username")
    password = st.text_input("Password", type="password", placeholder="Enter password")

    # Buttons in one row
    col1, col2 = st.columns(2)
    login_btn = col1.button("Login", use_container_width=True)
    reset_btn = col2.button("Reset Password", use_container_width=True)

    # Close HTML containers
    st.markdown("</div></div>", unsafe_allow_html=True)

    # RESET PASSWORD
    if reset_btn:
        st.session_state["go_reset_password"] = True
        st.session_state["reset_user"] = username.strip()
        st.rerun()

    # LOGIN
    if login_btn:
        if username.strip() == "" or password.strip() == "":
            st.error("Please enter both username and password.")
            return None, None

        row = get_user(username)
        if row is None:
            st.error("User not found.")
            return None, None

        if not verify_password(password, row[2]):
            st.error("Incorrect password.")
            return None, None

        st.session_state["authenticated"] = True
        st.session_state["username"] = username
        st.session_state["role"] = row[3]
        st.rerun()

    return None, None


