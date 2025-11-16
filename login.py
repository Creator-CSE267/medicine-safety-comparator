import streamlit as st
from user_database import get_user, verify_password
from password_reset import password_reset
from datetime import datetime
from styles import fix_login_spacing



# ---------------------------------------------------
# PROFESSIONAL LOGIN PAGE
# ---------------------------------------------------
def login_page():
    from styles import disable_all_background_for_login
disable_all_background_for_login()

    # PAGE FIX - remove all Streamlit default spacing
    st.markdown("""
        <style>
            /* REMOVE STREAMLIT TOP SPACING */
            .block-container {
                padding-top: 0 !important;
                margin-top: 0 !important;
            }
            header[data-testid="stHeader"] {
                display: none !important;
            }
            footer {
                visibility: hidden !important;
            }

            /* FULLSCREEN CENTER */
            .full-page {
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                height: 100vh;
                width: 100%;
            }

            /* LOGIN BOX */
            .login-card {
                width: 380px;
                padding: 35px;
                background: #1A1C23;
                border-radius: 18px;
                box-shadow: 0px 4px 25px rgba(0,0,0,0.35);
                text-align: center;
            }

            /* TITLE */
            .login-title {
                font-size: 28px;
                font-weight: 800;
                color: #2E86C1;
                margin-top: 10px;
                margin-bottom: 25px;
            }

            /* INPUT BOXES */
            .login-input input {
                background: #2A2C33 !important;
                color: white !important;
                border-radius: 10px !important;
                border: 1px solid #2E86C1 !important;
                height: 45px !important;
            }

            /* BUTTONS */
            .login-btn {
                width: 100%;
                background: #2E86C1 !important;
                color: white !important;
                border-radius: 8px !important;
                padding: 10px 0;
                font-size: 16px;
                font-weight: 600;
                margin-top: 5px;
            }
            .reset-btn {
                width: 100%;
                background: #444 !important;
                color: #fff !important;
                border-radius: 8px !important;
                padding: 10px 0;
                font-size: 16px;
                font-weight: 600;
                margin-top: 8px;
            }

        </style>
    """, unsafe_allow_html=True)

    # -----------------------
    # HTML WRAPPER START
    # -----------------------
    st.markdown("<div class='full-page'><div class='login-card'>", unsafe_allow_html=True)

    # LOGO
    st.image("logo.png", width=120)

    # TITLE
    st.markdown("<div class='login-title'>MedSafe Login</div>", unsafe_allow_html=True)

    # -----------------------
    # INPUTS
    # -----------------------
    username = st.text_input("Username", placeholder="Enter username")
    password = st.text_input("Password", type="password", placeholder="Enter password")

    # -----------------------
    # BUTTONS
    # -----------------------
    login_btn = st.button("Login", key="login_btn", help="Login", use_container_width=True)
    reset_btn = st.button("Reset Password", key="reset_btn", help="Change password", use_container_width=True)

    # -----------------------
    # HTML WRAPPER END
    # -----------------------
    st.markdown("</div></div>", unsafe_allow_html=True)

    # RESET PASSWORD BUTTON
    if reset_btn:
        st.session_state["go_reset_password"] = True
        st.session_state["reset_user"] = username.strip()
        st.rerun()

    # LOGIN ACTION
    if login_btn:
        if username.strip() == "" or password.strip() == "":
            st.error("Please enter both username and password.")
            return None, None

        row = get_user(username)
        if row is None:
            st.error("User not found.")
            return None, None

        stored_hash = row[2]
        role = row[3]

        if not verify_password(password, stored_hash):
            st.error("Incorrect password.")
            return None, None

        # SUCCESS
        st.session_state["authenticated"] = True
        st.session_state["username"] = username
        st.session_state["role"] = role
        st.session_state["last_active"] = datetime.now().isoformat()
        st.rerun()

    return None, None
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







