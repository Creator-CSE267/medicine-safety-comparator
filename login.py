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

    # ---------- STYLING ----------
    st.markdown("""
        <style>
            .login-container {
                width: 420px;
                margin: auto;
                margin-top: 120px;
                padding: 35px;
                background: white;
                border-radius: 18px;
                box-shadow: 0px 4px 25px rgba(0,0,0,0.25);
                text-align: center;
            }
            .login-title {
                font-size: 30px;
                font-weight: 800;
                color: #2E86C1;
                margin-bottom: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

    # ---------- LOGIN BOX ----------
    st.markdown("<div class='login-container'>", unsafe_allow_html=True)

    st.image("logo.png", width=140)
    st.markdown("<div class='login-title'>MedSafe Login</div>", unsafe_allow_html=True)

    username = st.text_input("Username", placeholder="Enter username")
    password = st.text_input("Password", type="password", placeholder="Enter password")

    col1, col2 = st.columns(2)
    login_btn = col1.button("Login")
    reset_btn = col2.button("Reset Password")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- RESET PASSWORD ----------
    if reset_btn:
        if username.strip() == "":
            st.error("Enter your username before resetting password.")
            return None, None

        st.session_state["go_reset_password"] = True
        st.session_state["reset_user"] = username.strip()

        st.rerun()
        return None, None   # stop execution

    # ---------- LOGIN PROCESS ----------
    if login_btn:

        # Empty fields
        if username.strip() == "" or password.strip() == "":
            st.error("Please enter both username and password.")
            return None, None

        # Fetch user
        row = get_user(username)
        if row is None:
            st.error("User not found.")
            return None, None

        stored_hash = row[2]
        role = row[3]

        # Wrong password
        if not verify_password(password, stored_hash):
            st.error("Incorrect password.")
            return None, None

        # ---------- LOGIN SUCCESS ----------
        st.session_state["authenticated"] = True
        st.session_state["username"] = username
        st.session_state["role"] = role
        st.session_state["last_active"] = datetime.now().isoformat()

        return username, role    # SUCCESS → return to app
        st.rerun()               # never reached

    return None, None
